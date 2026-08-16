import json
import os
import tempfile
import time
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_CONFIG_FILE = ROOT_DIR / "config.json"


class ConfigLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._created_root_config = False
        if not ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.write_text(json.dumps({"auth-key": "test-auth"}), encoding="utf-8")
            cls._created_root_config = True

        from services import config as config_module

        cls.config_module = config_module

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._created_root_config and ROOT_CONFIG_FILE.exists():
            ROOT_CONFIG_FILE.unlink()

    def test_load_settings_ignores_directory_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            data_dir = base_dir / "data"
            config_dir = base_dir / "config.json"
            os_auth_key = "env-auth"

            config_dir.mkdir()

            module = self.config_module
            old_base_dir = module.BASE_DIR
            old_data_dir = module.DATA_DIR
            old_config_file = module.CONFIG_FILE
            old_env_auth_key = module.os.environ.get("CHATGPT2API_AUTH_KEY")
            try:
                module.BASE_DIR = base_dir
                module.DATA_DIR = data_dir
                module.CONFIG_FILE = config_dir
                module.os.environ["CHATGPT2API_AUTH_KEY"] = os_auth_key

                settings = module._load_settings()

                self.assertEqual(settings.auth_key, os_auth_key)
                self.assertEqual(settings.refresh_account_interval_minute, 5)
            finally:
                module.BASE_DIR = old_base_dir
                module.DATA_DIR = old_data_dir
                module.CONFIG_FILE = old_config_file
                if old_env_auth_key is None:
                    module.os.environ.pop("CHATGPT2API_AUTH_KEY", None)
                else:
                    module.os.environ["CHATGPT2API_AUTH_KEY"] = old_env_auth_key


class ConfigCleanupTests(unittest.TestCase):
    """图片清理并发容错与触发式后台清理测试。"""

    def setUp(self):
        from services import config as config_module

        self.config_module = config_module

    def _make_store(self, tmp: Path):
        cfg_file = tmp / "config.json"
        cfg_file.write_text(json.dumps({
            "auth-key": "test-auth",
            "image_retention_hours": 4,
            "image_cleanup_batch_size": 50,
            "image_cleanup_batch_interval_secs": 0,
            "image_cleanup_max_batches_per_run": 10,
        }, ensure_ascii=False), encoding="utf-8")
        old_data_dir = self.config_module.DATA_DIR
        self.config_module.DATA_DIR = tmp
        self.addCleanup(setattr, self.config_module, "DATA_DIR", old_data_dir)
        store = self.config_module.ConfigStore(cfg_file)
        self.addCleanup(store.stop_cleanup_worker)
        return store

    def _create_old_images(self, images: Path, count: int = 10) -> None:
        images.mkdir(parents=True, exist_ok=True)
        old_ts = time.time() - 5 * 3600
        for i in range(count):
            p = images / f"old_{i}.png"
            p.write_bytes(b"x")
            os.utime(p, (old_ts, old_ts))

    def test_cleanup_old_images_tolerates_concurrent_deletion(self):
        """并发清理下部分文件已被其他线程删除时，清理不应抛 FileNotFoundError。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = self._make_store(tmp)
            images = tmp / "images"
            self._create_old_images(images, 10)
            # 模拟另一个清理线程已删除一半文件
            for p in list(images.glob("old_*.png"))[:5]:
                p.unlink()
            removed = store.cleanup_old_images()
            self.assertEqual(removed, 5)
            self.assertEqual(len(list(images.glob("old_*.png"))), 0)

    def test_request_cleanup_background_thread_deletes_old_images(self):
        """request_cleanup_old_images 应通过后台线程最终删除过期文件。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            store = self._make_store(tmp)
            images = tmp / "images"
            self._create_old_images(images, 5)
            store.request_cleanup_old_images()
            deadline = time.time() + 10
            while time.time() < deadline and list(images.glob("old_*.png")):
                time.sleep(0.05)
            self.assertEqual(len(list(images.glob("old_*.png"))), 0)


if __name__ == "__main__":
    unittest.main()
