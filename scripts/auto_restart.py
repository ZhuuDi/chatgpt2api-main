#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内存触发式自动重启脚本（服务器侧，配合 cron 使用）。

背景：高并发生图后，glibc malloc arena 会把历史峰值内存"停放"不归还 OS，
导致容器 RSS 长期停留在高位（曾见 7 天爬到 5GB）。本脚本在满足以下条件时
执行 docker restart chatgpt2api，清掉停放的高水位内存：

- 内存占用高：/metrics 的 runtime.rss_mb 超过 HIGH_MEM_MB（默认 3GB）
- 生图空闲：/metrics 的 generation.in_flight == 0 且连续 IDLE_CHECKS 次
  （每 5 分钟检查一次，3 次约 15 分钟无任务）
- 频率限制：24 小时内最多重启 1 次

部署（服务器 root 下）：
  cp scripts/auto_restart.py /opt/chatgpt2api-new/auto_restart.py
  chmod +x /opt/chatgpt2api-new/auto_restart.py
  ( crontab -l 2>/dev/null || true; \
    echo "*/5 * * * * /usr/bin/python3 /opt/chatgpt2api-new/auto_restart.py >> /opt/chatgpt2api-new/auto_restart.log 2>&1" ) | crontab -

前置：应用需已部署 M35（/metrics 暴露 generation.in_flight 与 runtime.rss_mb）。
"""
import json
import os
import subprocess
import time
import datetime

STATE = "/opt/chatgpt2api-new/.auto_restart_state.json"
METRICS = "http://127.0.0.1:3000/metrics"
HIGH_MEM_MB = 3000   # 内存阈值：超过 3GB 才考虑重启（可调整）
IDLE_CHECKS = 3      # 连续多少次空闲（每 5 分钟一次，3 次约 15 分钟）
MIN_INTERVAL = 24    # 24 小时内最多重启 1 次（小时）


def log(msg: str) -> None:
    print("%s %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def main() -> None:
    state: dict = {}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:
            state = {}
    now = time.time()
    if now - float(state.get("last_restart", 0)) < MIN_INTERVAL * 3600:
        return
    try:
        import urllib.request
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            d = json.loads(r.read().decode("utf-8"))
        inflight = int((d.get("generation") or {}).get("in_flight", -1))
        rss_mb = float((d.get("runtime") or {}).get("rss_mb", 0) or 0)
    except Exception as e:
        log("metrics 读取失败: %s" % e)
        return
    streak = int(state.get("idle_streak", 0))
    if rss_mb < HIGH_MEM_MB:
        # 内存不高，不需要重启，重置空闲计数
        state["idle_streak"] = 0
        state["last_check"] = now
        json.dump(state, open(STATE, "w"))
        return
    # 内存高：只有空闲时才累计
    streak = streak + 1 if inflight == 0 else 0
    state["idle_streak"] = streak
    state["last_check"] = now
    json.dump(state, open(STATE, "w"))
    log("rss_mb=%.0f in_flight=%d idle_streak=%d" % (rss_mb, inflight, streak))
    if streak >= IDLE_CHECKS:
        log("触发重启")
        subprocess.run(["docker", "restart", "chatgpt2api"])
        state["idle_streak"] = 0
        state["last_restart"] = time.time()
        json.dump(state, open(STATE, "w"))
        log("重启完成")


if __name__ == "__main__":
    main()
