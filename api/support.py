from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

from fastapi import HTTPException, Request

from services.account_service import account_service
from services.auth_service import auth_service
from services.config import config

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIST_DIR = BASE_DIR / "web_dist"


def extract_bearer_token(authorization: str | None) -> str:
    scheme, _, value = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return ""
    return value.strip()


def _legacy_admin_identity(token: str) -> dict[str, object] | None:
    auth_key = str(config.auth_key or "").strip()
    if auth_key and token == auth_key:
        return {"id": "admin", "name": "管理员", "role": "admin"}
    return None


def require_identity(authorization: str | None) -> dict[str, object]:
    token = extract_bearer_token(authorization)
    identity = _legacy_admin_identity(token) or auth_service.authenticate(token)
    if identity is None:
        raise HTTPException(status_code=401, detail={"error": "密钥无效或已失效，请重新登录"})
    return identity


def require_auth_key(authorization: str | None) -> None:
    require_identity(authorization)


def require_admin(authorization: str | None) -> dict[str, object]:
    identity = require_identity(authorization)
    if identity.get("role") != "admin":
        raise HTTPException(status_code=403, detail={"error": "需要管理员权限才能执行这个操作"})
    return identity


def resolve_image_base_url(request: Request) -> str:
    return config.base_url or f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"


def raise_image_quota_error(exc: Exception) -> None:
    message = str(exc)
    if "no available image quota" in message.lower():
        raise HTTPException(status_code=429, detail={"error": "no available image quota"}) from exc
    raise HTTPException(status_code=502, detail={"error": message}) from exc


def sanitize_cpa_pool(pool: dict | None) -> dict | None:
    if not isinstance(pool, dict):
        return None
    return {key: value for key, value in pool.items() if key != "secret_key"}


def sanitize_cpa_pools(pools: list[dict]) -> list[dict]:
    return [sanitized for pool in pools if (sanitized := sanitize_cpa_pool(pool)) is not None]


def sanitize_sub2api_server(server: dict | None) -> dict | None:
    if not isinstance(server, dict):
        return None
    sanitized = {key: value for key, value in server.items() if key not in {"password", "api_key"}}
    sanitized["has_api_key"] = bool(str(server.get("api_key") or "").strip())
    return sanitized


def sanitize_sub2api_servers(servers: list[dict]) -> list[dict]:
    return [sanitized for server in servers if (sanitized := sanitize_sub2api_server(server)) is not None]


def start_limited_account_watcher(stop_event: Event) -> Thread:
    # 账号探测改为"速率节流循环"：每个 tick 探测 rate*tick/60 个账号，
    # 账号冷却期（account_probe_min_interval_secs）防止短时间重复探测，
    # 新导入账号进入"未探测"队列按同一速率消化，不瞬间全量探测。
    tick_secs = max(1.0, float(config.account_probe_tick_secs))
    rate_per_minute = max(1, int(config.account_probe_rate_per_minute))
    batch_size = max(1, int(round(rate_per_minute * tick_secs / 60)))

    def worker() -> None:
        while not stop_event.is_set():
            try:
                tokens = account_service.list_refresh_candidates(batch_size)
                keepalive_tokens = account_service.list_refresh_token_keepalive_tokens()
                if tokens:
                    print(
                        "[account-watcher] probing "
                        f"{len(tokens)} accounts (rate {rate_per_minute}/min, tick {tick_secs}s)"
                    )
                    account_service.refresh_accounts(tokens)
                if keepalive_tokens:
                    print(f"[account-watcher] keepalive {len(keepalive_tokens)} refresh tokens")
                    result = account_service.keepalive_refresh_tokens(keepalive_tokens)
                    if result.get("errors"):
                        print(f"[account-watcher] keepalive errors: {result['errors']}")
            except Exception as exc:
                print(f"[account-watcher] fail {exc}")
            stop_event.wait(tick_secs)

    thread = Thread(target=worker, name="account-watcher", daemon=True)
    thread.start()
    return thread


def resolve_web_asset(requested_path: str) -> Path | None:
    if not WEB_DIST_DIR.exists():
        return None
    clean_path = requested_path.strip("/")
    base_dir = WEB_DIST_DIR.resolve()
    candidates = [base_dir / "index.html"] if not clean_path else [
        base_dir / Path(clean_path),
        base_dir / clean_path / "index.html",
        base_dir / f"{clean_path}.html",
    ]
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(base_dir)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


# 静态资源内存缓存：静态文件不会变化，首次读盘后驻留内存，
# 避免在 HDD/高并发写盘场景下每次请求都读盘导致响应被卡数十秒。
_WEB_ASSET_CACHE: dict[str, tuple[str, bytes]] = {}


def get_web_asset_content(requested_path: str) -> tuple[str, bytes] | None:
    """返回 web 静态资源 (文件名, 内容)。优先内存缓存，未命中再读盘并缓存。"""
    clean_path = requested_path.strip("/")
    key = clean_path or "index.html"
    cached = _WEB_ASSET_CACHE.get(key)
    if cached is not None:
        return cached
    path = resolve_web_asset(requested_path)
    if path is None:
        return None
    item = (path.name, path.read_bytes())
    # 控制缓存大小（防御性上限，避免异常路径无限增长）
    if len(_WEB_ASSET_CACHE) < 512:
        _WEB_ASSET_CACHE[key] = item
    return item
