from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class GenerationPoolFullError(RuntimeError):
    """图片生成队列已满：让下游降低并发或稍后重试。"""


class GenerationExecutor:
    """生图专用线程池：与网页 API 的 anyio 线程池完全分离。

    - max_workers：真正同时执行的生图任务数（image_generation_max_workers，默认 100）
    - queue_limit：在途（执行中 + 排队）请求上限（image_generation_queue_size，默认 300），
      超过直接快速失败（429），避免无限堆积拖垮服务器
    - 计数口径：1 个请求 = 1 个在途额度（流式续读不重复占额度），
      排队等待时间由调用方通过 timeout 计入单请求总预算
    """

    def __init__(self) -> None:
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._in_flight = 0

    @property
    def max_workers(self) -> int:
        try:
            from services.config import config
            return max(1, int(config.image_generation_max_workers or 100))
        except (TypeError, ValueError):
            return 100

    @property
    def queue_limit(self) -> int:
        try:
            from services.config import config
            return max(1, int(config.image_generation_queue_size or 300))
        except (TypeError, ValueError):
            return 300

    def _ensure_pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="image-gen",
                )
            return self._pool

    def acquire(self) -> None:
        """容量保护：在途请求达到队列上限时快速失败。"""
        with self._lock:
            if self._in_flight >= self.queue_limit:
                raise GenerationPoolFullError(
                    f"图片生成队列已满（上限 {self.queue_limit}，含排队），"
                    "请降低并发或稍后重试"
                )
            self._in_flight += 1

    def release(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    async def run(self, fn: Callable[..., Any], *args: Any, timeout: float | None = None) -> Any:
        """在生图专用池中执行 fn；timeout 用于兜底（排队等待计入单请求总预算）。"""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._ensure_pool(), fn, *args)
        if timeout is not None:
            return await asyncio.wait_for(future, timeout=timeout)
        return await future

    def shutdown(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=False)


generation_executor = GenerationExecutor()
