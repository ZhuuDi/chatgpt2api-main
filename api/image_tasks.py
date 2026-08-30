from __future__ import annotations

import re
import time as _time

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request
from services.image_task_service import image_task_service
from services.log_service import LoggedCall


class ImageGenerationTaskRequest(BaseModel):
    client_task_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    size: str | None = None
    quality: str = "auto"


class ResumePollRequest(BaseModel):
    extra_timeout_secs: float = Field(default=30.0, ge=5.0, le=120.0)


def _parse_task_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_image_ts(key: str, item) -> int | None:
    """从图片索引条目解析生成时间（文件名里的 10 位 unix 时间戳）。"""
    text = str(key or "")
    if isinstance(item, dict):
        text += "|" + str(item.get("name") or "") + "|" + str(item.get("path") or "")
    m = re.search(r"(\d{10})", text)
    return int(m.group(1)) if m else None


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/images/stats")
    async def image_stats(minutes: int = Query(default=30, ge=1, le=1440)):
        """对外公开：统计最近 N 分钟内生成的图片数（默认 30 分钟，minutes 可调）。

        基于图片索引（image_index.json）中每条记录的文件名时间戳统计，
        图片生成成功后写入索引，因此计数 = 该时间段内成功生成的图片数。
        """
        from services.image_storage_service import image_storage_service
        items = image_storage_service._load_clean_index() or {}
        cutoff = _time.time() - minutes * 60
        count = 0
        for key, item in items.items():
            ts = _parse_image_ts(key, item)
            if ts and ts >= cutoff:
                count += 1
        return {
            "ok": True,
            "minutes": minutes,
            "count": count,
            "since": _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(cutoff)),
            "now": _time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @router.get("/api/image-tasks")
    async def list_image_tasks(
        ids: str = Query(default=""),
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        return await run_in_threadpool(image_task_service.list_tasks, identity, _parse_task_ids(ids))

    @router.post("/api/image-tasks/generations")
    async def create_generation_task(
        body: ImageGenerationTaskRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/generations", body.model, "文生图任务", request_text=body.prompt), body.prompt)
        try:
            return await run_in_threadpool(
                image_task_service.submit_generation,
                identity,
                client_task_id=body.client_task_id,
                prompt=body.prompt,
                model=body.model,
                size=body.size,
                quality=body.quality,
                base_url=resolve_image_base_url(request),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/edits")
    async def create_edit_task(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        client_task_id = str(payload.get("client_task_id") or "").strip()
        if not client_task_id:
            raise HTTPException(status_code=400, detail={"error": "client_task_id is required"})
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/edits", model, "图生图任务", request_text=prompt), prompt)
        images = await read_image_sources(image_sources)
        masks = await read_image_sources(mask_sources) if mask_sources else None
        try:
            return await run_in_threadpool(
                image_task_service.submit_edit,
                identity,
                client_task_id=client_task_id,
                prompt=prompt,
                model=model,
                size=payload["size"],
                quality=payload["quality"],
                base_url=resolve_image_base_url(request),
                images=images,
                masks=masks,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/{task_id}/resume-poll")
    async def resume_image_poll(
        task_id: str,
        body: ResumePollRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        try:
            return await run_in_threadpool(
                image_task_service.resume_poll,
                identity,
                task_id,
                body.extra_timeout_secs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return router
