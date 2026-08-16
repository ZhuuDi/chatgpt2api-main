from __future__ import annotations

import threading

# 重 I/O 后台维护任务共享锁（图片清理 / 缩略图清理 / 备份）：
# 同一时刻只允许一个全盘扫描类任务运行，避免 HDD I/O 风暴叠加。
MAINTENANCE_LOCK = threading.Lock()

# 生图高峰判定阈值：在途生图请求 >= 该值时，后台维护任务主动避让
GENERATION_BUSY_THRESHOLD = 20


def generation_busy(threshold: int = GENERATION_BUSY_THRESHOLD) -> bool:
    """生图高峰期检测：在途（执行中 + 排队）生图请求数是否达到阈值。"""
    try:
        from services.generation_executor import generation_executor
        return generation_executor.in_flight() >= max(1, int(threshold))
    except Exception:
        return False
