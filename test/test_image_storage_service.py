from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from services.image_storage_service import ImageStorageService


def png_bytes(color=(255, 0, 0)) -> bytes:
    path = Path(tempfile.gettempdir()) / f"chatgpt2api-test-image-{color[0]}-{color[1]}-{color[2]}.png"
    Image.new("RGB", (2, 2), color=color).save(path, format="PNG")
    return path.read_bytes()


class FakeWebDAVClient:
    uploaded: dict[str, bytes] = {}
    deleted: list[str] = []

    def __init__(self, _settings):
        pass

    def put(self, rel: str, payload: bytes) -> str:
        self.uploaded[rel] = payload
        return f"https://dav.example.test/{rel}"

    def get(self, rel: str) -> bytes:
        return self.uploaded[rel]

    def delete(self, rel: str) -> bool:
        self.deleted.append(rel)
        self.uploaded.pop(rel, None)
        return True

    def test(self) -> dict[str, object]:
        self.put(".chatgpt2api_webdav_test.txt", b"chatgpt2api webdav test\n")
        self.delete(".chatgpt2api_webdav_test.txt")
        return {"ok": True, "status": 200, "error": None}


class ImageStorageServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)
        self.images_dir = self.data_dir / "images"
        self.settings = {
            "enabled": False,
            "mode": "local",
            "webdav_url": "",
            "webdav_username": "",
            "webdav_password": "",
            "webdav_root_path": "chatgpt2api/images",
            "public_base_url": "",
        }
        self.config_patcher = mock.patch("services.image_storage_service.config")
        self.mock_config = self.config_patcher.start()
        self.addCleanup(self.config_patcher.stop)
        self.mock_config.images_dir = self.images_dir
        self.mock_config.base_url = "http://app.test"
        self.mock_config.cleanup_old_images.return_value = 0
        self.mock_config.request_cleanup_old_images = mock.Mock()
        self.mock_config.get_image_storage_settings.side_effect = lambda: dict(self.settings)
        FakeWebDAVClient.uploaded = {}
        FakeWebDAVClient.deleted = []

    def service(self) -> ImageStorageService:
        svc = ImageStorageService(self.data_dir / "image_index.json")
        # 测试结束时立即落盘，避免延迟 flush 线程写入已删除的临时目录
        self.addCleanup(svc.flush)
        return svc

    def test_local_mode_saves_to_local_directory(self):
        stored = self.service().save(png_bytes(), "http://app.test")

        self.assertEqual(stored.storage, "local")
        self.assertTrue((self.images_dir / stored.rel).is_file())
        self.assertEqual(stored.url, f"http://app.test/images/{stored.rel}")
        self.mock_config.request_cleanup_old_images.assert_called_once()

    def test_webdav_mode_uploads_without_local_file(self):
        self.settings.update({
            "enabled": True,
            "mode": "webdav",
            "webdav_url": "https://dav.example.test",
            "webdav_password": "secret",
        })
        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            svc = self.service()
            stored = svc.save(png_bytes(), "http://app.test")
            svc.flush()
            payload = self.service().get_bytes(stored.rel)

        self.assertEqual(stored.storage, "webdav")
        self.assertFalse((self.images_dir / stored.rel).exists())
        self.assertIn(stored.rel, FakeWebDAVClient.uploaded)
        self.assertEqual(payload, FakeWebDAVClient.uploaded[stored.rel])

    def test_list_items_ignores_non_image_files(self):
        image = png_bytes()
        image_path = self.images_dir / "2026" / "05" / "07" / "sample.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image)
        (self.images_dir / ".DS_Store").write_text("not an image", encoding="utf-8")
        (self.images_dir / "2026" / ".DS_Store").write_text("not an image", encoding="utf-8")

        items = self.service().list_items("http://app.test")

        self.assertEqual([item["rel"] for item in items], ["2026/05/07/sample.png"])
        self.assertEqual(items[0]["storage"], "local")

    def test_both_mode_saves_to_local_and_webdav(self):
        self.settings.update({
            "enabled": True,
            "mode": "both",
            "webdav_url": "https://dav.example.test",
            "webdav_password": "secret",
            "public_base_url": "https://cdn.example.test/images",
        })
        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            stored = self.service().save(png_bytes(), "http://app.test")

        self.assertEqual(stored.storage, "both")
        self.assertTrue((self.images_dir / stored.rel).is_file())
        self.assertIn(stored.rel, FakeWebDAVClient.uploaded)
        self.assertEqual(stored.url, f"https://cdn.example.test/images/{stored.rel}")

    def test_test_webdav_writes_and_deletes_probe_file(self):
        self.settings.update({
            "enabled": True,
            "mode": "webdav",
            "webdav_url": "https://dav.example.test",
            "webdav_password": "secret",
        })
        with mock.patch("services.image_storage_service.WebDAVClient", FakeWebDAVClient):
            result = self.service().test_webdav()

        self.assertTrue(result["ok"])
        self.assertIn(".chatgpt2api_webdav_test.txt", FakeWebDAVClient.deleted)


    def test_save_then_flush_persists_index(self):
        """保存后立即 flush，索引文件应包含新条目（内存索引 + 合并落盘）。"""
        svc = self.service()
        stored = svc.save(png_bytes(), "http://app.test")
        svc.flush()
        index_file = self.data_dir / "image_index.json"
        self.assertTrue(index_file.exists())
        import json as _json
        data = _json.loads(index_file.read_text(encoding="utf-8"))
        self.assertIn(stored.rel, data.get("items", {}))
        # 再次保存（同实例，走内存索引，用不同颜色生成不同 md5）后条目数应为 2
        stored2 = svc.save(png_bytes(color=(0, 255, 0)), "http://app.test")
        svc.flush()
        data = _json.loads(index_file.read_text(encoding="utf-8"))
        self.assertEqual(len(data.get("items", {})), 2)
        self.assertIn(stored2.rel, data.get("items", {}))

    def test_remove_index_entries_updates_memory_and_disk(self):
        """清理删除文件后，remove_index_entries 同步移除内存与落盘条目。"""
        svc = self.service()
        stored = svc.save(png_bytes(), "http://app.test")
        removed = svc.remove_index_entries([stored.rel])
        self.assertEqual(removed, 1)
        # remove_index_entries 只移除索引条目，不删磁盘文件
        self.assertTrue(svc.has_local(stored.rel))
        self.assertNotIn(stored.rel, svc._load_clean_index())
        # 内存中已移除：再次删除返回 0
        self.assertEqual(svc.remove_index_entries([stored.rel]), 0)
        svc.flush()
        import json as _json
        data = _json.loads((self.data_dir / "image_index.json").read_text(encoding="utf-8"))
        self.assertNotIn(stored.rel, data.get("items", {}))


if __name__ == "__main__":
    unittest.main()
