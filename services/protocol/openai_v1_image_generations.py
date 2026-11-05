from __future__ import annotations

from typing import Any, Iterator

from services.protocol.conversation import (
    ConversationRequest,
    collect_image_outputs,
    count_text_tokens,
    stream_image_chunks,
    stream_image_outputs_with_pool,
)
from utils.animation import clamp_frame_count, clamp_frame_duration
from utils.image_tokens import count_image_output_items_tokens, image_usage


def handle(body: dict[str, Any]) -> dict[str, Any] | Iterator[dict[str, Any]]:
    prompt = str(body.get("prompt") or "")
    model = str(body.get("model") or "gpt-image-2")
    n = int(body.get("n") or 1)
    size = body.get("size")
    quality = str(body.get("quality") or "auto")
    response_format = str(body.get("response_format") or "b64_json")
    # Images 2.6 动图参数：animation=true 或 output_format=gif 时启用，帧数/时长仅动图语义有效
    is_animation = bool(body.get("animation")) or str(body.get("output_format") or "").lower() == "gif"
    output_format = str(body.get("output_format") or "").lower() or ("gif" if is_animation else "")
    frame_count = clamp_frame_count(body.get("frame_count")) if is_animation else 0
    frame_duration_ms = clamp_frame_duration(body.get("frame_duration_ms")) if is_animation else 0
    base_url = str(body.get("base_url") or "") or None
    progress_callback = body.get("progress_callback")
    outputs = stream_image_outputs_with_pool(ConversationRequest(
        prompt=prompt,
        model=model,
        n=n,
        size=size,
        quality=quality,
        response_format=response_format,
        animation=is_animation,
        output_format=output_format,
        frame_count=frame_count,
        frame_duration_ms=frame_duration_ms,
        base_url=base_url,
        message_as_error=True,
        progress_callback=progress_callback,
    ))
    if body.get("stream"):
        return stream_image_chunks(outputs)
    result = collect_image_outputs(outputs)
    result["usage"] = image_usage(
        input_text_tokens=count_text_tokens(prompt, model),
        output_tokens=count_image_output_items_tokens(result.get("data"), size, quality),
    )
    return result
