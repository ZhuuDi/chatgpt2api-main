# -*- coding: utf-8 -*-
"""animation.py — Images 2.6 GIF 动图参数支持

设计约定（2026-11-05 与需求方确认）：
- 2.6 是完整图像模型，动图意图由调用方在提示词中自然描述，面板**不注入**
  "GIF动图"这类制作要求（避免与用户意图冲突）；
- 帧数/帧时长参数：上游网页对话没有参数接口，提示词是唯一控制通道，因此
  这两个参数**会注入提示词**（帧数要求句式经实测上游精确执行）；
- 注入只影响概率，不构成合同：下载后用 Pillow 无损对齐——帧数不符时
  均匀抽帧/循环补帧，时长不符时重写每帧时长（等效调播放速度）；
- output_format=gif 的作用：过滤结果只保留动图资产（网页对话会额外产出
  分镜九宫格静帧）。

参数范围镜像官方 Images API：frame_count 2-64（默认 9），
frame_duration_ms 20-2000（默认 200，按 10ms 取整）。
"""
from __future__ import annotations

from io import BytesIO

FRAME_COUNT_MIN, FRAME_COUNT_MAX = 2, 64
FRAME_DURATION_MIN, FRAME_DURATION_MAX = 20, 2000

# 魔数嗅探表（按前缀匹配）
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8", "jpeg"),
    (b"GIF89a", "gif"),
    (b"GIF87a", "gif"),
)


def clamp_frame_count(value: object) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(FRAME_COUNT_MIN, min(FRAME_COUNT_MAX, n))


def clamp_frame_duration(value: object) -> int:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return 0
    # 官方语义：按 10ms 取整
    ms = max(FRAME_DURATION_MIN, min(FRAME_DURATION_MAX, ms))
    return (ms // 10) * 10


def sniff_image_format(data: bytes) -> str:
    """按魔数识别图片格式；无法识别时保守返回 png。"""
    for magic, fmt in _MAGIC_SIGNATURES:
        if data.startswith(magic):
            return fmt
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "png"


def is_animated_gif(data: bytes) -> bool:
    """是否为多帧 GIF 动图（单帧 GIF 视为静态图）。"""
    if sniff_image_format(data) != "gif":
        return False
    from PIL import Image

    with Image.open(BytesIO(data)) as im:
        return bool(getattr(im, "is_animated", False))


def image_dimensions(data: bytes) -> tuple[int, int]:
    """返回 (width, height)；解析失败返回 (0, 0)。"""
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def animation_prompt_prefix(frame_count: int = 0, frame_duration_ms: int = 0, intent: bool = True) -> str:
    """构造动图提示词前缀（固定放开头，用户提示词原文随后拼接）。

    - intent=True：包含「GIF动图」要求——调用方传 animation=true 时由面板代为
      表达动图意图，提示词无需再强调；
    - 帧数/帧时长条款：上游没有参数接口，写进提示词是唯一控制通道
      （帧数要求句式经实测可精确执行）。
    """
    parts: list[str] = []
    if intent:
        parts.append("GIF动图")
    if frame_count:
        parts.append(f"{frame_count}帧")
    if frame_duration_ms:
        parts.append(f"每帧{frame_duration_ms}毫秒")
    if not parts:
        return ""
    return "【制作要求：" + "，".join(parts) + "。】"


def align_gif(data: bytes, frame_count: int = 0, frame_duration_ms: int = 0) -> bytes:
    """把上游 GIF 对齐到 API 合同值（仅处理动图；静态单帧原样返回）。

    - 帧数多于请求值：按时间轴均匀抽帧；
    - 帧数少于请求值：循环拼接补齐（循环动画首尾衔接，视觉自然）；
    - 帧时长不符：无损重写每帧时长（Pillow 以 10ms 为存储单位）。
    """
    from PIL import Image, ImageSequence

    try:
        im = Image.open(BytesIO(data))
        if not bool(getattr(im, "is_animated", False)):
            return data
    except Exception:
        # 无法解析的数据原样返回（不因后处理失败而丢结果）
        return data

    frames: list[tuple[object, int]] = []
    for frame in ImageSequence.Iterator(im):
        duration = frame.info.get("duration")
        try:
            duration = int(duration) if duration else 100
        except (TypeError, ValueError):
            duration = 100
        frames.append((frame.copy(), max(20, duration)))

    # 帧数对齐
    if frame_count and len(frames) != frame_count:
        if len(frames) > frame_count:
            # 均匀抽样：按时间轴等距取 frame_count 帧
            idxs = {round(i * (len(frames) - 1) / (frame_count - 1)) for i in range(frame_count)}
            frames = [frames[i] for i in sorted(idxs)]
        else:
            base = list(frames)
            while len(frames) < frame_count:
                take = base[: min(len(base), frame_count - len(frames))]
                frames.extend(take)

    durations = []
    for _, origin_duration in frames:
        target = frame_duration_ms or origin_duration
        durations.append(max(20, (int(target) // 10) * 10))

    out_frames = [frame.convert("RGBA") for frame, _ in frames]
    buf = BytesIO()
    out_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=out_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()
