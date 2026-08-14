from __future__ import annotations

import sys

import uvicorn
from api import create_app

# 高并发下（生图任务/后台探测/网页请求共处单进程）降低 GIL 切换间隔：
# 默认 5ms 会让生图线程的 CPU 密集段长时间占住 GIL，事件循环（网页/API 请求）
# 被饿死数十秒；降到 1ms 后请求线程能频繁抢到 GIL，延迟从秒级降到亚秒级。
sys.setswitchinterval(0.001)

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, access_log=False, log_level="info")
