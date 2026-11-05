from __future__ import annotations

import unittest
from unittest import mock

from services.config import config
from services.openai_backend_api import (
    ImageRefusalError,
    OpenAIBackendAPI,
    is_image_edit_refusal,
)
from services.protocol import conversation as conversation_module
from services.protocol.conversation import (
    ConversationRequest,
    ImageOutput,
    _generate_single_image,
    stream_image_outputs,
)

# 上游 image_gen 工具对「误判为编辑已有图片」的固定英文拒绝句（实测原文）
EN_TOOL_REFUSAL = (
    "The user asked to edit or restore a specific existing image, but there is no usable image "
    "target in the conversation. Do not retry image generation. Tell the user to upload or "
    "identify the image before calling image_gen again."
)
# 模型改写的中文拒绝（实测原文，含「编辑已有图片 / 无法 / 抱歉」）
ZH_MODEL_REFUSAL = (
    "抱歉，目前无法直接生成这张图片，因为图像生成工具误将请求识别为“编辑已有图片”，"
    "而当前对话中没有可编辑的目标图片。\n\n请重新发送一次该请求（或开启一个新对话后发送）。"
)


def conversation_doc(*, user_text: str = "画一个角色立绘", refusal: str = "",
                     image_gen_task: bool = False, completed_text: str = "",
                     file_id: str = "") -> dict:
    """构造对话文档：user 消息 + 可选 tool 拒绝 + 可选图片任务 + 可选已完成 assistant 文本。"""
    mapping: dict = {
        "user": {
            "message": {
                "author": {"role": "user"},
                "create_time": 1,
                "content": {"content_type": "text", "parts": [user_text]},
                "metadata": {},
            }
        }
    }
    if refusal:
        mapping["tool"] = {
            "message": {
                "author": {"role": "tool"},
                "create_time": 2,
                "content": {"content_type": "text", "parts": [refusal]},
                "metadata": {"async_task_type": None},
            }
        }
    if image_gen_task:
        mapping["img"] = {
            "message": {
                "author": {"role": "tool"},
                "create_time": 3,
                "content": {
                    "content_type": "multimodal_text",
                    "parts": [{"content_type": "image_asset_pointer",
                               "asset_pointer": f"file-service://{file_id or 'file_000000001234567890abcdef12345678'}"}],
                },
                "metadata": {"async_task_type": "image_gen"},
            }
        }
    if completed_text:
        mapping["assistant"] = {
            "message": {
                "author": {"role": "assistant"},
                "create_time": 4,
                "content": {"content_type": "text", "parts": [completed_text]},
                "metadata": {"is_complete": True},
            }
        }
    return {"mapping": mapping}


class RefusalDetectionTests(unittest.TestCase):
    def test_english_tool_refusal_matches(self) -> None:
        self.assertTrue(is_image_edit_refusal(EN_TOOL_REFUSAL))

    def test_chinese_model_refusal_matches(self) -> None:
        self.assertTrue(is_image_edit_refusal(ZH_MODEL_REFUSAL))

    def test_plain_text_reply_does_not_match(self) -> None:
        self.assertFalse(is_image_edit_refusal("这是一张温暖房间里的元气问候图片。"))

    def test_content_policy_text_does_not_match(self) -> None:
        self.assertFalse(is_image_edit_refusal("非常抱歉，该提示可能违反了我们的内容政策。"))

    def test_find_refusal_in_conversation(self) -> None:
        doc = conversation_doc(refusal=EN_TOOL_REFUSAL)
        self.assertTrue(OpenAIBackendAPI._find_image_refusal_in_conversation(doc))

    def test_find_refusal_absent(self) -> None:
        doc = conversation_doc(image_gen_task=True)
        self.assertFalse(OpenAIBackendAPI._find_image_refusal_in_conversation(doc))

    def test_turn_ended_without_image_detects_completed_text(self) -> None:
        doc = conversation_doc(completed_text=EN_TOOL_REFUSAL)
        self.assertTrue(OpenAIBackendAPI._conversation_turn_ended_without_image(doc))

    def test_turn_ended_ignores_doc_with_image_task(self) -> None:
        doc = conversation_doc(image_gen_task=True, completed_text="done")
        self.assertFalse(OpenAIBackendAPI._conversation_turn_ended_without_image(doc))

    def test_turn_ended_ignores_incomplete_text(self) -> None:
        doc = conversation_doc(refusal=EN_TOOL_REFUSAL)
        self.assertFalse(OpenAIBackendAPI._conversation_turn_ended_without_image(doc))


class PollRefusalTests(unittest.TestCase):
    """轮询阶段命中拒绝特征应立即抛 ImageRefusalError，而不是等到超时。"""

    def setUp(self) -> None:
        self._saved = dict(config.data)
        config.data = dict(
            self._saved,
            image_poll_interval_secs=0.01,
            image_poll_initial_wait_secs=0.01,
            image_check_before_hit_enabled=False,
            image_settle_enabled=False,
            image_refusal_early_exit_enabled=False,
        )

    def tearDown(self) -> None:
        config.data = self._saved

    def _poll(self, doc: dict):
        backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)

        class _DummySession:
            pass

        with mock.patch.object(OpenAIBackendAPI, "_get_conversation", return_value=doc), \
             mock.patch.object(OpenAIBackendAPI, "_query_backend_tasks", return_value=[]):
            return backend._poll_image_results("conv-refusal", timeout_secs=5.0)

    def test_poll_raises_refusal_within_one_attempt(self) -> None:
        doc = conversation_doc(refusal=EN_TOOL_REFUSAL)
        with self.assertRaises(ImageRefusalError):
            self._poll(doc)

    def test_poll_returns_file_ids_when_image_exists(self) -> None:
        doc = conversation_doc(image_gen_task=True)
        file_ids, sediment_ids = self._poll(doc)
        self.assertTrue(file_ids)

    def test_poll_dry_run_does_not_raise_for_unknown_ending(self) -> None:
        """兜底检测默认 dry-run：命中只打日志，仍继续轮询直至超时。"""
        doc = conversation_doc(completed_text="本回合仅返回了说明文本，没有触发图片工具。")
        with self.assertRaises(Exception) as ctx:
            self._poll(doc)
        self.assertNotIsInstance(ctx.exception, ImageRefusalError)


class StreamRefusalTests(unittest.TestCase):
    """SSE 文本本身就是拒绝时应立即上抛，不再进入轮询。"""

    def setUp(self) -> None:
        self._saved = dict(config.data)
        config.data = dict(self._saved, image_total_timeout_secs=30)

    def tearDown(self) -> None:
        config.data = self._saved

    def _stream(self, text: str, turn_use_case: str = "image gen"):
        backend = mock.MagicMock()
        events = [
            {"type": "conversation.delta", "delta": text, "conversation_id": "conv-x"},
            {"type": "conversation.done", "conversation_id": "conv-x", "text": text,
             "file_ids": [], "sediment_ids": [], "turn_use_case": turn_use_case,
             "tool_invoked": True, "blocked": False},
        ]
        with mock.patch.object(conversation_module, "conversation_events", return_value=iter(events)):
            return list(stream_image_outputs(
                backend,
                ConversationRequest(prompt="画一个角色立绘", model="gpt-image-2"),
            ))

    def test_stream_raises_on_zh_refusal(self) -> None:
        with self.assertRaises(ImageRefusalError):
            self._stream(ZH_MODEL_REFUSAL)

    def test_stream_raises_on_en_refusal(self) -> None:
        with self.assertRaises(ImageRefusalError):
            self._stream(EN_TOOL_REFUSAL)

    def test_stream_message_unchanged_for_normal_text(self) -> None:
        # 普通文本回复不触发拒绝早退，原样作为 message 输出
        outputs = self._stream("这是一张温暖房间里的元气问候图片。", turn_use_case="")
        self.assertTrue(any(item.kind == "message" for item in outputs))


class RefusalRetryTests(unittest.TestCase):
    """_generate_single_image：拒绝应换账号重试并追加澄清词，而不是直接失败。"""

    def setUp(self) -> None:
        self._saved = dict(config.data)
        config.data = dict(
            self._saved,
            image_total_timeout_secs=120,
            image_retry_budget=5,
        )

    def tearDown(self) -> None:
        config.data = self._saved

    def test_retry_then_success(self) -> None:
        attempts: list[ConversationRequest] = []

        def fake_stream(backend, request, index, total):
            attempts.append(request)
            if len(attempts) == 1:
                raise ImageRefusalError(EN_TOOL_REFUSAL, "conv-1")
            yield ImageOutput(kind="result", model=request.model, index=index, total=total,
                              data=[{"url": "http://example.test/ok.png"}], conversation_id="conv-2")

        backend = mock.MagicMock()
        with mock.patch.object(conversation_module.account_service, "get_available_access_token", return_value="tok-1"), \
             mock.patch.object(conversation_module.account_service, "get_account", return_value={"email": "a@test"}), \
             mock.patch.object(conversation_module.account_service, "mark_image_result"), \
             mock.patch.object(conversation_module.backend_pool, "acquire", return_value=backend), \
             mock.patch.object(conversation_module.backend_pool, "release"), \
             mock.patch.object(conversation_module, "stream_image_outputs", side_effect=fake_stream), \
             mock.patch.object(conversation_module, "_remove_image_conversation_later"):
            outputs = _generate_single_image(
                ConversationRequest(prompt="画一个角色立绘", model="gpt-image-2"), 1, 1)

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].retry_prompt_suffix, "")
        # 重试轮必须带上澄清词，直指误判来源
        self.assertIn("全新的图片生成请求", attempts[1].retry_prompt_suffix)
        self.assertTrue(any(item.kind == "result" for item in outputs))

    def test_retry_exhausted_raises_with_code(self) -> None:
        calls = {"n": 0}

        def fake_stream(backend, request, index, total):
            calls["n"] += 1
            raise ImageRefusalError(EN_TOOL_REFUSAL, "conv-x")
            yield  # pragma: no cover

        backend = mock.MagicMock()
        with mock.patch.object(conversation_module.account_service, "get_available_access_token", return_value="tok-1"), \
             mock.patch.object(conversation_module.account_service, "get_account", return_value={"email": "a@test"}), \
             mock.patch.object(conversation_module.account_service, "mark_image_result"), \
             mock.patch.object(conversation_module.backend_pool, "acquire", return_value=backend), \
             mock.patch.object(conversation_module.backend_pool, "release"), \
             mock.patch.object(conversation_module, "stream_image_outputs", side_effect=fake_stream), \
             mock.patch.object(conversation_module, "_remove_image_conversation_later"):
            with self.assertRaises(conversation_module.ImageGenerationError) as ctx:
                _generate_single_image(
                    ConversationRequest(prompt="画一个角色立绘", model="gpt-image-2"), 1, 1)

        self.assertEqual(ctx.exception.code, "upstream_edit_refusal")
        # 3 次重试 + 1 次首轮 = 4 次尝试后放弃
        self.assertEqual(calls["n"], 4)


if __name__ == "__main__":
    unittest.main()
