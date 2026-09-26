from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService, _no_image_quota_message
from services.config import config
from services.protocol.conversation import is_upload_quota_error
from services.storage.json_storage import JSONStorageBackend


REAL_UPLOAD_429_ERROR = (
    '/backend-api/files failed: status=429, body={"detail": {"code": "throttled", '
    '"error_code": "throttled", "message": "你已达到文件上传上限。请1天 内重试。", '
    '"type": "throttled"}}'
)


def _limits_progress(remaining: int) -> list[dict]:
    return [
        {"feature_name": "image_gen", "remaining": 5},
        {"feature_name": "file_upload", "remaining": remaining},
    ]


class UploadRemainingTests(unittest.TestCase):
    def test_upload_remaining_parses_limits_progress(self) -> None:
        self.assertEqual(
            AccountService._upload_remaining({"limits_progress": _limits_progress(15)}),
            15,
        )
        self.assertEqual(
            AccountService._upload_remaining({"limits_progress": [{"feature_name": "file_upload", "remaining": 0}]}),
            0,
        )

    def test_upload_remaining_returns_none_when_unknown(self) -> None:
        self.assertIsNone(AccountService._upload_remaining({}))
        self.assertIsNone(AccountService._upload_remaining({"limits_progress": []}))
        self.assertIsNone(AccountService._upload_remaining({"limits_progress": [{"feature_name": "image_gen"}]}))
        self.assertIsNone(
            AccountService._upload_remaining(
                {"limits_progress": [{"feature_name": "file_upload", "remaining": "bad"}]}
            )
        )
        self.assertIsNone(AccountService._upload_remaining(None))


class UploadAvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_min_remaining = config.data.get("image_upload_min_remaining")
        self._original_throttle_hours = config.data.get("image_upload_throttle_hours")

    def tearDown(self) -> None:
        if self._original_min_remaining is None:
            config.data.pop("image_upload_min_remaining", None)
        else:
            config.data["image_upload_min_remaining"] = self._original_min_remaining
        if self._original_throttle_hours is None:
            config.data.pop("image_upload_throttle_hours", None)
        else:
            config.data["image_upload_throttle_hours"] = self._original_throttle_hours

    def _service(self) -> AccountService:
        return AccountService(JSONStorageBackend(Path(self._tmp_dir) / "accounts.json"))

    def test_threshold_filters_low_remaining_but_keeps_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as self._tmp_dir:
            service = self._service()
            high = {"status": "正常", "quota": 5, "limits_progress": _limits_progress(78)}
            low = {"status": "正常", "quota": 5, "limits_progress": _limits_progress(3)}
            unknown = {"status": "正常", "quota": 5}

            self.assertTrue(service._is_upload_available(high))
            self.assertFalse(service._is_upload_available(low))
            # remaining 未知（未探测/旧数据）放行，由生图时的 429 兜底标记补上
            self.assertTrue(service._is_upload_available(unknown))

    def test_zero_threshold_disables_prefilter(self) -> None:
        config.data["image_upload_min_remaining"] = 0
        with tempfile.TemporaryDirectory() as self._tmp_dir:
            service = self._service()
            low = {"status": "正常", "quota": 5, "limits_progress": _limits_progress(0)}
            self.assertTrue(service._is_upload_available(low))

    def test_restore_marker_blocks_until_expired(self) -> None:
        with tempfile.TemporaryDirectory() as self._tmp_dir:
            service = self._service()
            account = {
                "status": "正常",
                "quota": 5,
                "limits_progress": _limits_progress(78),
                "upload_restore_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            }
            self.assertFalse(service._is_upload_available(account))

            expired = dict(account)
            expired["upload_restore_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            self.assertTrue(service._is_upload_available(expired))


class UploadQuotaErrorClassifierTests(unittest.TestCase):
    def test_matches_real_upstream_error(self) -> None:
        self.assertTrue(is_upload_quota_error(REAL_UPLOAD_429_ERROR))

    def test_ignores_other_errors(self) -> None:
        self.assertFalse(
            is_upload_quota_error(
                '/backend-api/conversation failed: status=429, body={"detail":{"code":"throttled"}}'
            )
        )
        self.assertFalse(
            is_upload_quota_error(
                '/backend-api/files failed: status=500, body={"detail": "internal error"}'
            )
        )
        self.assertFalse(is_upload_quota_error("curl: (28) Operation timed out"))
        self.assertFalse(is_upload_quota_error(""))


class MarkUploadThrottledTests(unittest.TestCase):
    def _assert_restore_hours(self, restore_at: str, expected_hours: int) -> None:
        delta = datetime.fromisoformat(restore_at) - datetime.now(timezone.utc)
        self.assertLess(abs(delta - timedelta(hours=expected_hours)).total_seconds(), 120)

    def test_parses_upstream_duration_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{"access_token": "token-1", "status": "正常", "quota": 2}])

            updated = service.mark_upload_throttled("token-1", REAL_UPLOAD_429_ERROR.replace("请1天", "请23小时"))
            self.assertIsNotNone(updated)
            self._assert_restore_hours(updated["upload_restore_at"], 23)

            updated = service.mark_upload_throttled("token-1", "请2天 内重试。")
            self._assert_restore_hours(updated["upload_restore_at"], 48)

    def test_falls_back_to_configured_default_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{"access_token": "token-1", "status": "正常", "quota": 2}])

            updated = service.mark_upload_throttled("token-1", "unrelated error")
            self._assert_restore_hours(updated["upload_restore_at"], config.image_upload_throttle_hours)

    def test_marked_account_is_skipped_for_upload_requests_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{"access_token": "token-1", "status": "正常", "quota": 5}])
            service.fetch_remote_info = lambda access_token, *args, **kwargs: service.get_account(access_token)

            service.mark_upload_throttled("token-1", REAL_UPLOAD_429_ERROR)

            with self.assertRaises(RuntimeError):
                service.get_available_access_token(needs_upload=True)

            # 纯文生图不受上传标记影响
            token = service.get_available_access_token()
            service.release_image_slot(token)
            self.assertEqual(token, "token-1")


class NeedsUploadSelectionTests(unittest.TestCase):
    def _service_with_accounts(self, tmp_dir: str):
        service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
        service.add_account_items(
            [
                {
                    "access_token": "token-low",
                    "status": "正常",
                    "quota": 5,
                    "limits_progress": _limits_progress(3),
                },
                {
                    "access_token": "token-high",
                    "status": "正常",
                    "quota": 5,
                    "limits_progress": _limits_progress(78),
                },
            ]
        )
        service.fetch_remote_info = lambda access_token, *args, **kwargs: service.get_account(access_token)
        return service

    def test_needs_upload_prefers_accounts_with_remaining_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self._service_with_accounts(tmp_dir)

            token = service.get_available_access_token(needs_upload=True)
            service.release_image_slot(token)
            self.assertEqual(token, "token-high")

    def test_upload_requests_still_round_robin_when_prefilter_disabled(self) -> None:
        original = config.data.get("image_upload_min_remaining")
        config.data["image_upload_min_remaining"] = 0
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = self._service_with_accounts(tmp_dir)

                seen = set()
                for _ in range(6):
                    token = service.get_available_access_token(needs_upload=True)
                    seen.add(token)
                    service.release_image_slot(token)
                self.assertEqual(seen, {"token-low", "token-high"})
        finally:
            if original is None:
                config.data.pop("image_upload_min_remaining", None)
            else:
                config.data["image_upload_min_remaining"] = original

    def test_non_upload_requests_keep_using_low_remaining_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self._service_with_accounts(tmp_dir)

            seen = set()
            for _ in range(6):
                token = service.get_available_access_token()
                seen.add(token)
                service.release_image_slot(token)
            self.assertEqual(seen, {"token-low", "token-high"})

    def test_all_exhausted_error_keeps_image_quota_substring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {
                        "access_token": "token-low",
                        "status": "正常",
                        "quota": 5,
                        "limits_progress": _limits_progress(3),
                    }
                ]
            )
            service.fetch_remote_info = lambda access_token, *args, **kwargs: service.get_account(access_token)

            with self.assertRaises(RuntimeError) as ctx:
                service.get_available_access_token(needs_upload=True)

            message = str(ctx.exception).lower()
            # api/support.py raise_image_quota_error 依赖该子串映射 HTTP 429
            self.assertIn("no available image quota", message)
            self.assertIn("file upload quota", message)

    def test_no_image_quota_message_variants(self) -> None:
        self.assertEqual(_no_image_quota_message(None, None, False), "no available image quota")
        self.assertEqual(_no_image_quota_message("plus", None, False), "no available plus image quota")
        self.assertIn("no available image quota", _no_image_quota_message(None, None, True))


class UploadableQuotaSummaryTests(unittest.TestCase):
    def test_summary_excludes_upload_unavailable_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    # 上传可用：计入 total_quota
                    {"access_token": "t-ok", "status": "正常", "quota": 7, "limits_progress": _limits_progress(78)},
                    # 余量低于阈值：生图可用但上传不可用 → excluded
                    {"access_token": "t-low", "status": "正常", "quota": 2, "limits_progress": _limits_progress(3)},
                    # 无探测数据（余量未知放行），但有 429 标记 → excluded
                    {"access_token": "t-marked", "status": "正常", "quota": 4},
                    # 限流账号：生图本就不可用，不计入 excluded
                    {"access_token": "t-limited", "status": "限流", "quota": 0},
                ]
            )
            service.mark_upload_throttled("t-marked", REAL_UPLOAD_429_ERROR)

            summary = service.summarize_uploadable_image_quota()

            self.assertEqual(summary["total_quota"], 7)
            self.assertEqual(summary["uploadable_accounts"], 1)
            self.assertEqual(summary["excluded_accounts"], 2)
            self.assertEqual(summary["excluded_quota"], 6)

    def test_summary_zero_threshold_counts_all_image_available_accounts(self) -> None:
        original = config.data.get("image_upload_min_remaining")
        config.data["image_upload_min_remaining"] = 0
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items(
                    [
                        {"access_token": "t-ok", "status": "正常", "quota": 7, "limits_progress": _limits_progress(78)},
                        {"access_token": "t-zero", "status": "正常", "quota": 3, "limits_progress": _limits_progress(0)},
                    ]
                )

                summary = service.summarize_uploadable_image_quota()

                self.assertEqual(summary["total_quota"], 10)
                self.assertEqual(summary["uploadable_accounts"], 2)
                self.assertEqual(summary["excluded_accounts"], 0)
                self.assertEqual(summary["excluded_quota"], 0)
        finally:
            if original is None:
                config.data.pop("image_upload_min_remaining", None)
            else:
                config.data["image_upload_min_remaining"] = original


if __name__ == "__main__":
    unittest.main()
