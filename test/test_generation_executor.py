from __future__ import annotations

import unittest
from unittest import mock

from services.generation_executor import GenerationExecutor, GenerationPoolFullError


class GenerationExecutorTests(unittest.IsolatedAsyncioTestCase):
    """生图专用线程池：容量保护、配置读取与执行测试。"""

    def setUp(self):
        self.executor = GenerationExecutor()
        self.addCleanup(self.executor.shutdown)

    async def test_run_executes_function_in_pool(self):
        result = await self.executor.run(lambda: 42)
        self.assertEqual(result, 42)

    async def test_acquire_release_counts(self):
        self.assertEqual(self.executor.in_flight(), 0)
        self.executor.acquire()
        self.assertEqual(self.executor.in_flight(), 1)
        self.executor.release()
        self.assertEqual(self.executor.in_flight(), 0)

    async def test_queue_full_raises(self):
        with mock.patch("services.config.config") as cfg:
            cfg.image_generation_queue_size = 2
            self.executor.acquire()
            self.executor.acquire()
            with self.assertRaises(GenerationPoolFullError):
                self.executor.acquire()

    async def test_max_workers_reads_config(self):
        with mock.patch("services.config.config") as cfg:
            cfg.image_generation_max_workers = 7
            self.assertEqual(self.executor.max_workers, 7)

    async def test_queue_limit_reads_config(self):
        with mock.patch("services.config.config") as cfg:
            cfg.image_generation_queue_size = 77
            self.assertEqual(self.executor.queue_limit, 77)


if __name__ == "__main__":
    unittest.main()
