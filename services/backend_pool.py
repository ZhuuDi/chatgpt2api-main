from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI


class OpenAIBackendPool:
    """账号级 OpenAIBackendAPI / curl_cffi Session 常驻池。

    每个账号最多保留 image_account_concurrency 个可复用实例，
    避免每个图片请求/每次重试都新建 Session 导致文件描述符与连接暴涨。

    - acquire：池中有空闲实例则复用，否则新建；
    - release：归还实例，超过每账号上限则关闭丢弃；
    - invalidate：token 失效/轮换时关闭该账号全部实例，防止复用失效连接；
    - 空闲实例由后台线程定期回收（默认 300s），避免账号池过大时长期占用 FD。
    """

    def __init__(self, idle_timeout_secs: float = 300.0) -> None:
        self._idle_timeout_secs = idle_timeout_secs
        self._pool: dict[str, deque[OpenAIBackendAPI]] = {}
        self._last_used: dict[int, float] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._cleaner = threading.Thread(
            target=self._cleanup_loop,
            name="backend-pool-cleaner",
            daemon=True,
        )
        self._cleaner.start()

    def _limit(self) -> int:
        try:
            return max(1, int(config.image_account_concurrency or 1))
        except (TypeError, ValueError):
            return 2

    def acquire(self, token: str) -> OpenAIBackendAPI:
        """获取账号 token 对应的后端客户端实例。"""
        with self._lock:
            items = self._pool.get(token)
            if items:
                backend = items.popleft()
                self._last_used.pop(id(backend), None)
                return backend
        return OpenAIBackendAPI(access_token=token)

    def release(self, token: str, backend: OpenAIBackendAPI | None) -> None:
        """归还实例；池已关闭或超过每账号上限时关闭丢弃。"""
        if backend is None:
            return
        with self._lock:
            if self._closed:
                backend.close()
                return
            items = self._pool.setdefault(token, deque())
            if len(items) >= self._limit():
                backend.close()
                return
            items.append(backend)
            self._last_used[id(backend)] = time.time()

    def invalidate(self, token: str) -> None:
        """token 失效或轮换时，关闭该账号在池中的所有实例。"""
        with self._lock:
            items = self._pool.pop(token, None)
            if items:
                for backend in items:
                    self._last_used.pop(id(backend), None)
        if items:
            for backend in items:
                try:
                    backend.close()
                except Exception:
                    pass

    def account_count(self) -> int:
        """池中当前持有实例的账号数量（用于指标观测）。"""
        with self._lock:
            return len(self._pool)

    def close_idle(self, idle_secs: float | None = None) -> int:
        """关闭空闲超过阈值的实例，返回关闭数量。"""
        idle_secs = self._idle_timeout_secs if idle_secs is None else idle_secs
        now = time.time()
        expired: list[tuple[str, OpenAIBackendAPI]] = []
        with self._lock:
            for token, items in list(self._pool.items()):
                keep: deque[OpenAIBackendAPI] = deque()
                for backend in items:
                    last = self._last_used.get(id(backend), now)
                    if now - last >= idle_secs:
                        expired.append((token, backend))
                    else:
                        keep.append(backend)
                if keep:
                    self._pool[token] = keep
                else:
                    self._pool.pop(token, None)
        for _token, backend in expired:
            try:
                backend.close()
            except Exception:
                pass
        return len(expired)

    def _cleanup_loop(self) -> None:
        while not self._closed:
            time.sleep(60.0)
            try:
                self.close_idle()
            except Exception:
                pass

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            items = [backend for q in self._pool.values() for backend in q]
            self._pool.clear()
            self._last_used.clear()
        for backend in items:
            try:
                backend.close()
            except Exception:
                pass


backend_pool = OpenAIBackendPool()
