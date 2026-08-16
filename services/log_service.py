from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from services.config import DATA_DIR, config
from services.generation_executor import GenerationPoolFullError, generation_executor
from services.protocol.error_response import anthropic_error_response, openai_error_response
from utils.helper import anthropic_sse_stream, sse_json_stream

LOG_TYPE_CALL = "call"
LOG_TYPE_ACCOUNT = "account"
INTERNAL_RESPONSE_KEYS = {"_account_email", "_conversation_id"}


class LogService:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 批量缓冲写入：add() 只追加到内存缓冲，后台线程每 0.1s flush 一次，
        # 或缓冲达到 1MB 时立即 flush，避免高并发下每个日志条目都 open/close 文件。
        self._lock = threading.Lock()
        self._buffer: list[str] = []
        self._buffer_bytes = 0
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="log-flush",
            daemon=True,
        )
        self._flush_thread.start()

    @staticmethod
    def _legacy_id(raw_line: str, line_number: int) -> str:
        payload = f"{line_number}:{raw_line}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:24]

    def _parse_line(self, raw_line: str, line_number: int) -> dict[str, Any] | None:
        try:
            item = json.loads(raw_line)
        except Exception:
            return None
        if not isinstance(item, dict):
            return None
        parsed = dict(item)
        parsed["id"] = str(parsed.get("id") or self._legacy_id(raw_line, line_number))
        return parsed

    @staticmethod
    def _serialize_item(item: dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _matches_filters(item: dict[str, Any], *, type: str = "", start_date: str = "", end_date: str = "") -> bool:
        t = str(item.get("time") or "")
        day = t[:10]
        if type and item.get("type") != type:
            return False
        if start_date and day < start_date:
            return False
        if end_date and day > end_date:
            return False
        return True

    def add(self, type: str, summary: str = "", detail: dict[str, Any] | None = None, **data: Any) -> None:
        item = {
            "id": uuid4().hex,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": type,
            "summary": summary,
            "detail": detail or data,
        }
        line = self._serialize_item(item) + "\n"
        force = False
        with self._lock:
            self._buffer.append(line)
            self._buffer_bytes += len(line.encode("utf-8", errors="ignore"))
            if self._buffer_bytes >= 1024 * 1024:
                force = True
        if force:
            self._flush()

    def _flush_loop(self) -> None:
        while True:
            time.sleep(0.5)
            try:
                self._flush()
            except Exception:
                pass

    def _flush(self) -> None:
        """将缓冲日志批量写入文件，并按大小轮转。"""
        with self._lock:
            if not self._buffer:
                return
            lines = self._buffer
            self._buffer = []
            self._buffer_bytes = 0
        try:
            with self.path.open("a", encoding="utf-8") as file:
                file.write("".join(lines))
            self._rotate_if_needed()
        except Exception:
            # 写入失败时恢复缓冲，避免丢日志
            with self._lock:
                self._buffer = lines + self._buffer
                self._buffer_bytes += sum(len(line.encode("utf-8", errors="ignore")) for line in lines)

    def flush(self) -> None:
        """立即将缓冲日志落盘（读取/退出前调用）。"""
        self._flush()

    def _rotate_if_needed(self) -> None:
        try:
            max_bytes = max(1024 * 1024, int(config.log_max_bytes))
            backup_count = max(1, int(config.log_backup_count))
        except (TypeError, ValueError):
            max_bytes = 100 * 1024 * 1024
            backup_count = 5
        try:
            if not self.path.exists() or self.path.stat().st_size < max_bytes:
                return
            oldest = self.path.with_suffix(self.path.suffix + f".{backup_count}")
            if oldest.exists():
                oldest.unlink()
            for i in range(backup_count - 1, 0, -1):
                src = self.path.with_suffix(self.path.suffix + f".{i}")
                if src.exists():
                    src.replace(self.path.with_suffix(self.path.suffix + f".{i + 1}"))
            if self.path.exists():
                self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        except Exception:
            pass

    def _read_tail_lines(self, max_bytes: int | None = None) -> list[str]:
        """从文件尾部读取最近若干行，避免全量加载大日志。

        - max_bytes=None：读取整个文件（兼容小文件与兜底）；
        - 否则只读文件末尾最多 max_bytes 字节，首行若被截断则丢弃，
          保证返回的都是完整行。
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        size = self.path.stat().st_size
        if max_bytes is None or size <= max_bytes:
            return self.path.read_text(encoding="utf-8").splitlines()
        with self.path.open("rb") as f:
            f.seek(size - max_bytes)
            data = f.read()
        lines = data.decode("utf-8", "replace").splitlines()
        if lines:
            lines = lines[1:]  # 丢弃可能被截断的首行
        return lines

    def list(self, type: str = "", start_date: str = "", end_date: str = "", limit: int = 200) -> list[dict[str, Any]]:
        self.flush()
        if not self.path.exists():
            return []
        # 优先只读尾部（5MB），若尾部解析不出条目（如文件末尾为异常数据），
        # 逐级扩大读取范围直到找到条目或读完整份文件。
        for max_bytes in (5 * 1024 * 1024, 20 * 1024 * 1024, None):
            lines = self._read_tail_lines(max_bytes)
            items: list[dict[str, Any]] = []
            for line_number in range(len(lines) - 1, -1, -1):
                item = self._parse_line(lines[line_number], line_number)
                if item is None:
                    continue
                if not self._matches_filters(item, type=type, start_date=start_date, end_date=end_date):
                    continue
                items.append(item)
                if len(items) >= limit:
                    return items
            if items:
                return items
        return []

    def delete(self, ids: list[str]) -> dict[str, int]:
        self.flush()
        target_ids = {str(item or "").strip() for item in ids if str(item or "").strip()}
        if not self.path.exists() or not target_ids:
            return {"removed": 0}
        lines = self.path.read_text(encoding="utf-8").splitlines()
        kept_lines: list[str] = []
        removed = 0
        for line_number, raw_line in enumerate(lines):
            item = self._parse_line(raw_line, line_number)
            if item is None:
                kept_lines.append(raw_line)
                continue
            if str(item.get("id") or "") in target_ids:
                removed += 1
                continue
            kept_lines.append(self._serialize_item(item))
        content = "\n".join(kept_lines)
        if content:
            content += "\n"
        self.path.write_text(content, encoding="utf-8")
        return {"removed": removed}


log_service = LogService(DATA_DIR / "logs.jsonl")


def _collect_urls(value: object) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str):
                urls.append(item)
            elif key == "urls" and isinstance(item, list):
                urls.extend(str(url) for url in item if isinstance(url, str))
            else:
                urls.extend(_collect_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls(item))
    return urls


def _collect_account_emails(value: object) -> list[str]:
    emails: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"_account_email", "account_email"} and isinstance(item, str) and item.strip():
                emails.append(item.strip())
            else:
                emails.extend(_collect_account_emails(item))
    elif isinstance(value, list):
        for item in value:
            emails.extend(_collect_account_emails(item))
    return emails


def _collect_conversation_ids(value: object) -> list[str]:
    ids: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "_conversation_id" and isinstance(item, str) and item.strip():
                ids.append(item.strip())
            else:
                ids.extend(_collect_conversation_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.extend(_collect_conversation_ids(item))
    return ids


def _strip_internal_response_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_internal_response_fields(item)
            for key, item in value.items()
            if key not in INTERNAL_RESPONSE_KEYS
        }
    if isinstance(value, list):
        return [_strip_internal_response_fields(item) for item in value]
    return value


def _request_excerpt(text: object, limit: int = 1000) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _image_error_response(exc: Exception) -> JSONResponse:
    from services.protocol.conversation import public_image_error_message

    message = public_image_error_message(str(exc))
    if "no available image quota" in message.lower():
        return openai_error_response(
            {
                "error": {
                    "message": "no available image quota",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
            429,
        )
    if hasattr(exc, "to_openai_error") and hasattr(exc, "status_code"):
        return JSONResponse(status_code=int(exc.status_code), content=exc.to_openai_error())
    return openai_error_response(message, 502)


def _protocol_error_response(exc: Exception, status_code: int, sse: str) -> JSONResponse:
    message = str(exc)
    if sse == "anthropic":
        return anthropic_error_response(message, status_code)
    return openai_error_response(message, status_code)


def _next_item(items):
    try:
        return True, next(items)
    except StopIteration:
        return False, None


@dataclass
class LoggedCall:
    identity: dict[str, object]
    endpoint: str
    model: str
    summary: str
    started: float = field(default_factory=time.time)
    request_text: str = ""
    request_shape: dict[str, int] | None = None

    async def run(self, handler, *args, sse: str = "openai"):
        from services.protocol.conversation import ImageGenerationError

        # 生图请求走独立执行器，与网页 API 的 anyio 线程池完全分离；
        # 非生图请求保持原有 run_in_threadpool 行为
        is_image = self.endpoint.startswith("/v1/images")
        acquired = False
        if is_image:
            try:
                generation_executor.acquire()
                acquired = True
            except GenerationPoolFullError as exc:
                self.log("调用失败", status="failed", error=str(exc))
                return openai_error_response({
                    "error": {
                        "message": str(exc),
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "image_generation_queue_full",
                    }
                }, 429)
        try:
            try:
                if is_image:
                    # 排队等待也计入单请求总预算（total + 30s 兜底），避免客户端超时后服务端仍无限排队
                    result = await generation_executor.run(
                        handler, *args, timeout=float(config.image_total_timeout_secs) + 30.0
                    )
                else:
                    result = await run_in_threadpool(handler, *args)
            except ImageGenerationError as exc:
                self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""),
                         conversation_id=getattr(exc, "conversation_id", ""))
                return _image_error_response(exc)
            except HTTPException as exc:
                self.log("调用失败", status="failed", error=str(exc.detail))
                raise
            except Exception as exc:
                self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""))
                if self.endpoint.startswith("/v1/images"):
                    return _image_error_response(exc)
                return _protocol_error_response(exc, 502, sse)

            if isinstance(result, dict):
                self.log("调用完成", result)
                response = dict(result)
                response.pop("_account_email", None)
                return response

            sender = anthropic_sse_stream if sse == "anthropic" else sse_json_stream
            try:
                if is_image:
                    has_first, first = await generation_executor.run(_next_item, result)
                else:
                    has_first, first = await run_in_threadpool(_next_item, result)
            except ImageGenerationError as exc:
                self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""),
                         conversation_id=getattr(exc, "conversation_id", ""))
                return _image_error_response(exc)
            except HTTPException as exc:
                self.log("调用失败", status="failed", error=str(exc.detail))
                raise
            except Exception as exc:
                self.log("调用失败", status="failed", error=str(exc), account_email=getattr(exc, "account_email", ""))
                if self.endpoint.startswith("/v1/images"):
                    return _image_error_response(exc)
                return _protocol_error_response(exc, 502, sse)
            if not has_first:
                self.log("流式调用结束")
                return StreamingResponse(sender(()), media_type="text/event-stream")

            if is_image:
                async def _stream_async():
                    nonlocal acquired
                    try:
                        items = self.stream(itertools.chain([first], result))
                        while True:
                            has_next, item = await generation_executor.run(_next_item, items)
                            if not has_next:
                                break
                            yield sender(item)
                    finally:
                        # 流式结束后才归还生图池额度
                        if acquired:
                            acquired = False
                            generation_executor.release()

                return StreamingResponse(_stream_async(), media_type="text/event-stream")

            return StreamingResponse(sender(self.stream(itertools.chain([first], result))), media_type="text/event-stream")
        finally:
            if acquired:
                generation_executor.release()

    def stream(self, items):
        urls: list[str] = []
        account_emails: list[str] = []
        conversation_ids: list[str] = []
        failed = False
        try:
            for item in items:
                urls.extend(_collect_urls(item))
                account_emails.extend(_collect_account_emails(item))
                conversation_ids.extend(_collect_conversation_ids(item))
                yield _strip_internal_response_fields(item)
        except Exception as exc:
            failed = True
            self.log(
                "流式调用失败",
                status="failed",
                error=str(exc),
                urls=urls,
                account_email=(account_emails[0] if account_emails else getattr(exc, "account_email", "")),
                conversation_id=(conversation_ids[0] if conversation_ids else getattr(exc, "conversation_id", "")),
            )
            if self.endpoint.startswith("/v1/images") and not hasattr(exc, "to_openai_error"):
                from services.protocol.conversation import ImageGenerationError, public_image_error_message

                raise ImageGenerationError(public_image_error_message(str(exc))) from exc
            raise
        finally:
            if not failed:
                self.log("流式调用结束", urls=urls, account_email=account_emails[0] if account_emails else "",
                         conversation_id=conversation_ids[0] if conversation_ids else "")

    def log(self, suffix: str, result: object = None, status: str = "success", error: str = "",
            urls: list[str] | None = None, account_email: str = "", conversation_id: str = "") -> None:
        detail = {
            "key_id": self.identity.get("id"),
            "key_name": self.identity.get("name"),
            "role": self.identity.get("role"),
            "endpoint": self.endpoint,
            "model": self.model,
            "started_at": datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_ms": int((time.time() - self.started) * 1000),
            "status": status,
        }
        request_excerpt = _request_excerpt(self.request_text)
        if request_excerpt:
            detail["request_text"] = request_excerpt
        if self.request_shape:
            detail["request_shape"] = self.request_shape
        if error:
            detail["error"] = error
        email = str(account_email or "").strip()
        if not email:
            emails = _collect_account_emails(result)
            email = emails[0] if emails else ""
        if email:
            detail["account_email"] = email
        conv_id = str(conversation_id or "").strip()
        if not conv_id:
            conv_ids = _collect_conversation_ids(result)
            conv_id = conv_ids[0] if conv_ids else ""
        if conv_id:
            detail["conversation_id"] = conv_id
        collected_urls = [*(urls or []), *_collect_urls(result)]
        if collected_urls and not self.endpoint.startswith("/v1/search"):
            detail["urls"] = list(dict.fromkeys(collected_urls))
        log_service.add(LOG_TYPE_CALL, f"{self.summary}{suffix}", detail)
