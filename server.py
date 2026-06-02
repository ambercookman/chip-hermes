import asyncio
import base64
import html
import json
import os
import re
import secrets
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(f"Generated admin password: {ADMIN_PASSWORD}")

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
ENV_FILE_PATH = Path(HERMES_HOME) / ".env"
CONFIG_YAML_PATH = Path(HERMES_HOME) / "config.yaml"
PAIRING_DIR = Path(HERMES_HOME) / "pairing"
CODE_TTL_SECONDS = 3600

# Maps provider env-var key → (hermes provider name, default base_url)
# Order determines priority when multiple keys are set.
PROVIDER_CONFIG: list[tuple[str, str, str]] = [
    ("OPENROUTER_API_KEY", "openrouter", "https://openrouter.ai/api/v1"),
    ("DEEPSEEK_API_KEY",   "deepseek",   "https://api.deepseek.com/v1"),
    ("DASHSCOPE_API_KEY",  "alibaba",    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("GLM_API_KEY",        "zai",        "https://api.z.ai/api/paas/v4"),
    ("KIMI_API_KEY",       "kimi-coding","https://api.moonshot.ai/v1"),
    ("MINIMAX_API_KEY",    "minimax",    "https://api.minimax.io/anthropic"),
    ("HF_TOKEN",           "huggingface","https://router.huggingface.co/v1"),
]

# Registry of known Hermes env vars exposed in the UI.
# Each entry: (key, label, category, is_password)
ENV_VAR_DEFS = [
    # Model
    ("LLM_MODEL", "Model", "model", False),
    # Providers
    ("OPENROUTER_API_KEY", "OpenRouter API Key", "provider", True),
    ("DEEPSEEK_API_KEY", "DeepSeek API Key", "provider", True),
    ("DASHSCOPE_API_KEY", "DashScope API Key", "provider", True),
    ("GLM_API_KEY", "GLM / Z.AI API Key", "provider", True),
    ("KIMI_API_KEY", "Kimi API Key", "provider", True),
    ("MINIMAX_API_KEY", "MiniMax API Key", "provider", True),
    ("HF_TOKEN", "Hugging Face Token", "provider", True),
    # Tools
    ("PARALLEL_API_KEY", "Parallel API Key", "tool", True),
    ("FIRECRAWL_API_KEY", "Firecrawl API Key", "tool", True),
    ("TAVILY_API_KEY", "Tavily API Key", "tool", True),
    ("FAL_KEY", "FAL API Key", "tool", True),
    ("BROWSERBASE_API_KEY", "Browserbase API Key", "tool", True),
    ("BROWSERBASE_PROJECT_ID", "Browserbase Project ID", "tool", False),
    ("GITHUB_TOKEN", "GitHub Token", "tool", True),
    ("VOICE_TOOLS_OPENAI_KEY", "OpenAI Voice Key", "tool", True),
    ("HONCHO_API_KEY", "Honcho API Key", "tool", True),
    # Messaging — Telegram
    ("TELEGRAM_BOT_TOKEN", "Telegram Bot Token", "messaging", True),
    ("TELEGRAM_ALLOWED_USERS", "Telegram Allowed Users", "messaging", False),
    # Messaging — Discord
    ("DISCORD_BOT_TOKEN", "Discord Bot Token", "messaging", True),
    ("DISCORD_ALLOWED_USERS", "Discord Allowed Users", "messaging", False),
    # Messaging — Slack
    ("SLACK_BOT_TOKEN", "Slack Bot Token", "messaging", True),
    ("SLACK_APP_TOKEN", "Slack App Token", "messaging", True),
    # Messaging — WhatsApp
    ("WHATSAPP_ENABLED", "WhatsApp Enabled", "messaging", False),
    # Messaging — Email
    ("EMAIL_ADDRESS", "Email Address", "messaging", False),
    ("EMAIL_PASSWORD", "Email Password", "messaging", True),
    ("EMAIL_IMAP_HOST", "Email IMAP Host", "messaging", False),
    ("EMAIL_SMTP_HOST", "Email SMTP Host", "messaging", False),
    # Messaging — Mattermost
    ("MATTERMOST_URL", "Mattermost URL", "messaging", False),
    ("MATTERMOST_TOKEN", "Mattermost Token", "messaging", True),
    # Messaging — Matrix
    ("MATRIX_HOMESERVER", "Matrix Homeserver", "messaging", False),
    ("MATRIX_ACCESS_TOKEN", "Matrix Access Token", "messaging", True),
    ("MATRIX_USER_ID", "Matrix User ID", "messaging", False),
    # Messaging — General
    ("GATEWAY_ALLOW_ALL_USERS", "Allow All Users", "messaging", False),
]

PASSWORD_KEYS = {key for key, _, _, is_pw in ENV_VAR_DEFS if is_pw}

PROVIDER_KEYS = [key for key, _, cat, _ in ENV_VAR_DEFS if cat == "provider" and key != "LLM_MODEL"]
CHANNEL_KEYS = {
    "Telegram": "TELEGRAM_BOT_TOKEN",
    "Discord": "DISCORD_BOT_TOKEN",
    "Slack": "SLACK_BOT_TOKEN",
    "WhatsApp": "WHATSAPP_ENABLED",
    "Email": "EMAIL_ADDRESS",
    "Mattermost": "MATTERMOST_TOKEN",
    "Matrix": "MATRIX_ACCESS_TOKEN",
}


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def write_env_file(path: Path, env_vars: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    categories = {"model": "Model", "provider": "Providers", "tool": "Tools", "messaging": "Messaging"}
    grouped: dict[str, list[str]] = {cat: [] for cat in categories}
    known_keys = {key for key, _, _, _ in ENV_VAR_DEFS}
    key_to_cat = {key: cat for key, _, cat, _ in ENV_VAR_DEFS}

    for key, value in env_vars.items():
        if not value:
            continue
        cat = key_to_cat.get(key, "other")
        line = f"{key}={value}"
        if cat in grouped:
            grouped[cat].append(line)
        else:
            grouped.setdefault("other", []).append(line)

    lines = []
    for cat, heading in categories.items():
        entries = grouped.get(cat, [])
        if entries:
            lines.append(f"# {heading}")
            lines.extend(sorted(entries))
            lines.append("")

    other = grouped.get("other", [])
    if other:
        lines.append("# Other")
        lines.extend(sorted(other))
        lines.append("")

    path.write_text("\n".join(lines) + "\n" if lines else "")


def write_hermes_yaml_config(env_vars: dict[str, str]):
    """Write model block to config.yaml. Hermes no longer reads LLM_MODEL from .env."""
    CONFIG_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if CONFIG_YAML_PATH.exists():
        try:
            existing = yaml.safe_load(CONFIG_YAML_PATH.read_text()) or {}
        except Exception:
            existing = {}

    # Detect first provider with a key set
    active_provider: str | None = None
    active_base_url: str | None = None
    for env_key, pname, purl in PROVIDER_CONFIG:
        if env_vars.get(env_key):
            active_provider = pname
            active_base_url = purl
            break

    llm_model = env_vars.get("LLM_MODEL", "")

    # Update only the keys we manage; preserve everything else in model:
    model_block: dict = {}
    if isinstance(existing.get("model"), dict):
        model_block = dict(existing["model"])

    if active_provider:
        model_block["provider"] = active_provider
        model_block["base_url"] = active_base_url
    if llm_model:
        model_block["default"] = llm_model

    if model_block:
        existing["model"] = model_block

    CONFIG_YAML_PATH.write_text(
        yaml.dump(existing, default_flow_style=False, allow_unicode=True)
    )


def mask_secrets(env_vars: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in env_vars.items():
        if key in PASSWORD_KEYS and value:
            result[key] = value[:8] + "***" if len(value) > 8 else "***"
        else:
            result[key] = value
    return result


def merge_secrets(new_vars: dict[str, str], existing_vars: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in new_vars.items():
        if key in PASSWORD_KEYS and value.endswith("***"):
            result[key] = existing_vars.get(key, "")
        else:
            result[key] = value
    return result


class BasicAuthBackend(AuthenticationBackend):
    async def authenticate(self, conn):
        if "Authorization" not in conn.headers:
            return None

        auth = conn.headers["Authorization"]
        try:
            scheme, credentials = auth.split()
            if scheme.lower() != "basic":
                return None
            decoded = base64.b64decode(credentials).decode("ascii")
        except (ValueError, UnicodeDecodeError):
            raise AuthenticationError("Invalid credentials")

        username, _, password = decoded.partition(":")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return AuthCredentials(["authenticated"]), SimpleUser(username)

        raise AuthenticationError("Invalid credentials")


def require_auth(request: Request):
    if not request.user.is_authenticated:
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="hermes"'},
        )
    return None


class GatewayManager:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = deque(maxlen=500)
        self.start_time: float | None = None
        self.restart_count = 0
        self._read_tasks: list[asyncio.Task] = []

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        self.state = "starting"
        try:
            env = os.environ.copy()
            env["HERMES_HOME"] = HERMES_HOME
            env_vars = read_env_file(ENV_FILE_PATH)
            env.update(env_vars)

            self.process = await asyncio.create_subprocess_exec(
                "hermes", "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.start_time = time.time()
            task = asyncio.create_task(self._read_output())
            self._read_tasks.append(task)
        except Exception as e:
            self.state = "error"
            self.logs.append(f"Failed to start gateway: {e}")

    async def stop(self):
        if not self.process or self.process.returncode is not None:
            self.state = "stopped"
            return
        self.state = "stopping"
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.state = "stopped"
        self.start_time = None

    async def restart(self):
        await self.stop()
        self.restart_count += 1
        await self.start()

    async def _read_output(self):
        try:
            while self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                cleaned = ANSI_ESCAPE.sub("", decoded)
                self.logs.append(cleaned)
        except asyncio.CancelledError:
            return
        if self.process and self.process.returncode is not None and self.state == "running":
            self.state = "error"
            self.logs.append(f"Gateway exited with code {self.process.returncode}")

    def get_status(self) -> dict:
        pid = None
        if self.process and self.process.returncode is None:
            pid = self.process.pid
        uptime = None
        if self.start_time and self.state == "running":
            uptime = int(time.time() - self.start_time)
        return {
            "state": self.state,
            "pid": pid,
            "uptime": uptime,
            "restart_count": self.restart_count,
        }


gateway = GatewayManager()
config_lock = asyncio.Lock()


async def homepage(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    return templates.TemplateResponse(request, "index.html")


async def health(request: Request):
    return JSONResponse({"status": "ok", "gateway": gateway.state})


async def api_config_get(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    async with config_lock:
        env_vars = read_env_file(ENV_FILE_PATH)
    defs = [
        {"key": key, "label": label, "category": cat, "password": is_pw}
        for key, label, cat, is_pw in ENV_VAR_DEFS
    ]
    return JSONResponse({"vars": mask_secrets(env_vars), "defs": defs})


async def api_config_put(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        restart = body.pop("_restartGateway", False)
        new_vars = body.get("vars", {})

        async with config_lock:
            existing = read_env_file(ENV_FILE_PATH)
            merged = merge_secrets(new_vars, existing)
            # Preserve any existing vars not in the UI
            for key, value in existing.items():
                if key not in merged:
                    merged[key] = value
            write_env_file(ENV_FILE_PATH, merged)
            write_hermes_yaml_config(merged)

        if restart:
            asyncio.create_task(gateway.restart())

        return JSONResponse({"ok": True, "restarting": restart})
    except Exception as e:
        print(f"Config save error: {type(e).__name__}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_status(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    env_vars = read_env_file(ENV_FILE_PATH)

    providers = {}
    for key in PROVIDER_KEYS:
        label = key.replace("_API_KEY", "").replace("_TOKEN", "").replace("HF_", "HuggingFace ").replace("_", " ").title()
        providers[label] = {"configured": bool(env_vars.get(key))}

    channels = {}
    for name, key in CHANNEL_KEYS.items():
        val = env_vars.get(key, "")
        channels[name] = {"configured": bool(val) and val.lower() not in ("false", "0", "no")}

    return JSONResponse({
        "gateway": gateway.get_status(),
        "providers": providers,
        "channels": channels,
    })


async def api_logs(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    return JSONResponse({"lines": list(gateway.logs)})


async def api_gateway_start(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    asyncio.create_task(gateway.start())
    return JSONResponse({"ok": True})


async def api_gateway_stop(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    asyncio.create_task(gateway.stop())
    return JSONResponse({"ok": True})


async def api_gateway_restart(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    asyncio.create_task(gateway.restart())
    return JSONResponse({"ok": True})


def _load_pairing_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_pairing_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _pairing_platforms(suffix: str) -> list[str]:
    if not PAIRING_DIR.exists():
        return []
    return [
        f.stem.rsplit(f"-{suffix}", 1)[0]
        for f in PAIRING_DIR.glob(f"*-{suffix}.json")
    ]


async def api_pairing_pending(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    now = time.time()
    results = []
    for platform in _pairing_platforms("pending"):
        pending = _load_pairing_json(PAIRING_DIR / f"{platform}-pending.json")
        for code, info in pending.items():
            age = now - info.get("created_at", now)
            if age > CODE_TTL_SECONDS:
                continue
            results.append({
                "platform": platform,
                "code": code,
                "user_id": info.get("user_id", ""),
                "user_name": info.get("user_name", ""),
                "age_minutes": int(age / 60),
            })
    return JSONResponse({"pending": results})


async def api_pairing_approve(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    platform = body.get("platform", "")
    code = body.get("code", "").upper().strip()
    if not platform or not code:
        return JSONResponse({"error": "platform and code required"}, status_code=400)

    pending_path = PAIRING_DIR / f"{platform}-pending.json"
    pending = _load_pairing_json(pending_path)
    if code not in pending:
        return JSONResponse({"error": "Code not found or expired"}, status_code=404)

    entry = pending.pop(code)
    _save_pairing_json(pending_path, pending)

    approved_path = PAIRING_DIR / f"{platform}-approved.json"
    approved = _load_pairing_json(approved_path)
    approved[entry["user_id"]] = {
        "user_name": entry.get("user_name", ""),
        "approved_at": time.time(),
    }
    _save_pairing_json(approved_path, approved)

    return JSONResponse({"ok": True, "user_id": entry["user_id"], "user_name": entry.get("user_name", "")})


async def api_pairing_deny(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    platform = body.get("platform", "")
    code = body.get("code", "").upper().strip()
    if not platform or not code:
        return JSONResponse({"error": "platform and code required"}, status_code=400)

    pending_path = PAIRING_DIR / f"{platform}-pending.json"
    pending = _load_pairing_json(pending_path)
    if code in pending:
        del pending[code]
        _save_pairing_json(pending_path, pending)

    return JSONResponse({"ok": True})


async def api_pairing_approved(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    results = []
    for platform in _pairing_platforms("approved"):
        approved = _load_pairing_json(PAIRING_DIR / f"{platform}-approved.json")
        for user_id, info in approved.items():
            results.append({
                "platform": platform,
                "user_id": user_id,
                "user_name": info.get("user_name", ""),
                "approved_at": info.get("approved_at", 0),
            })
    return JSONResponse({"approved": results})


async def telegram_webhook_proxy(request: Request):
    """Forward Telegram webhook POSTs to the hermes gateway's internal webhook server.

    Railway only exposes one port (PORT/8080). The hermes gateway starts its own
    webhook listener on TELEGRAM_WEBHOOK_PORT (default 8443). This route receives
    Telegram's HTTPS POST on port 8080 and forwards it to the internal listener so
    webhook mode works without exposing a second port.
    """
    webhook_port = int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443"))
    body = await request.body()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{webhook_port}/telegram",
                content=body,
                headers={
                    "content-type": request.headers.get("content-type", "application/json"),
                    "x-telegram-bot-api-secret-token": request.headers.get(
                        "x-telegram-bot-api-secret-token", ""
                    ),
                },
                timeout=10.0,
            )
        return PlainTextResponse(resp.text, status_code=resp.status_code)
    except Exception as e:
        return PlainTextResponse(f"Webhook proxy error: {e}", status_code=502)


async def vapi_webhook(request: Request):
    """Receive Vapi server messages; send Telegram summary for end-of-call reports."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    print(f"[vapi-webhook] {json.dumps(body)}")

    message = body.get("message") or {}
    if message.get("type") != "end-of-call-report":
        return JSONResponse({"ok": True})

    analysis = message.get("analysis") or {}
    call = message.get("call") or {}
    artifact = message.get("artifact") or {}
    customer = call.get("customer") or {}

    summary = analysis.get("summary") or "No summary available."
    success_eval = analysis.get("successEvaluation")
    call_id = call.get("id", "")
    phone_number = customer.get("number", "unknown")
    ended_reason = message.get("endedReason", "unknown")
    costs = call.get("costs")
    transcript_msgs = artifact.get("messages") or []

    # Duration from startedAt / endedAt
    duration_str = ""
    try:
        started_at = call.get("startedAt") or ""
        ended_at = call.get("endedAt") or ""
        if started_at and ended_at:
            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            secs = int((end - start).total_seconds())
            duration_str = f"{secs // 60}m {secs % 60}s"
    except Exception:
        pass

    # Cost total
    cost_str = ""
    try:
        if isinstance(costs, list):
            total = sum(c.get("cost", 0) for c in costs if isinstance(c, dict))
            cost_str = f"${total:.4f}"
        elif isinstance(costs, dict):
            total = costs.get("total")
            if total is not None:
                cost_str = f"${float(total):.4f}"
    except Exception:
        pass

    e = html.escape
    parts = [f"📞 <b>Vapi Call Report</b>"]
    parts.append(f"<b>To:</b> <code>{e(phone_number)}</code>")
    if call_id:
        parts.append(f"<b>Call ID:</b> <code>{e(call_id)}</code>")
    parts.append(f"<b>Ended:</b> {e(ended_reason)}")
    if duration_str:
        parts.append(f"<b>Duration:</b> {duration_str}")
    if success_eval is not None:
        parts.append(f"<b>Outcome:</b> {e(str(success_eval))}")
    if cost_str:
        parts.append(f"<b>Cost:</b> {cost_str}")
    parts.append("")
    parts.append(f"<b>Summary:</b>\n{e(summary)}")
    if transcript_msgs:
        parts.append(f"\n<i>{len(transcript_msgs)} messages in transcript</i>")

    tg_text = "\n".join(parts)

    env_vars = read_env_file(ENV_FILE_PATH)
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or env_vars.get("TELEGRAM_BOT_TOKEN", "")
    allowed_users = os.environ.get("TELEGRAM_ALLOWED_USERS") or env_vars.get("TELEGRAM_ALLOWED_USERS", "")

    if not bot_token or not allowed_users:
        print("[vapi-webhook] Telegram not configured, skipping notification")
        return JSONResponse({"ok": True})

    chat_id = allowed_users.split(",")[0].strip()
    if not chat_id:
        print("[vapi-webhook] No chat ID in TELEGRAM_ALLOWED_USERS")
        return JSONResponse({"ok": True})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": tg_text, "parse_mode": "HTML"},
                timeout=10.0,
            )
        if resp.status_code != 200:
            print(f"[vapi-webhook] Telegram error {resp.status_code}: {resp.text}")
        else:
            print(f"[vapi-webhook] Telegram notification sent to {chat_id}")
    except Exception as exc:
        print(f"[vapi-webhook] Telegram send failed: {exc}")

    return JSONResponse({"ok": True})


async def api_vapi_call(request: Request):
    """Trigger an outbound Vapi call. Requires admin auth."""
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    required_fields = ["phone_number", "caller_name", "call_purpose", "call_purpose_short", "opening_line"]
    missing = [f for f in required_fields if not body.get(f)]
    if missing:
        return JSONResponse({"error": f"Missing required fields: {', '.join(missing)}"}, status_code=400)

    vapi_api_key = os.environ.get("VAPI_API_KEY", "")
    vapi_assistant_id = os.environ.get("VAPI_ASSISTANT_ID", "")
    vapi_phone_number_id = os.environ.get("VAPI_PHONE_NUMBER_ID", "")

    if not vapi_api_key:
        return JSONResponse({"error": "VAPI_API_KEY not configured"}, status_code=500)
    if not vapi_assistant_id:
        return JSONResponse({"error": "VAPI_ASSISTANT_ID not configured"}, status_code=500)
    if not vapi_phone_number_id:
        return JSONResponse({"error": "VAPI_PHONE_NUMBER_ID not configured"}, status_code=500)

    variable_values = {
        "caller_name": body["caller_name"],
        "call_purpose": body["call_purpose"],
        "call_purpose_short": body["call_purpose_short"],
        "questions": body.get("questions") or "N/A",
        "appointment_details": body.get("appointment_details") or "N/A",
        "opening_line": body["opening_line"],
    }
    if body.get("callback_number"):
        variable_values["callback_number"] = body["callback_number"]

    vapi_payload = {
        "assistantId": vapi_assistant_id,
        "phoneNumberId": vapi_phone_number_id,
        "customer": {"number": body["phone_number"]},
        "assistantOverrides": {"variableValues": variable_values},
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.vapi.ai/call",
                json=vapi_payload,
                headers={"Authorization": f"Bearer {vapi_api_key}"},
                timeout=30.0,
            )
        try:
            resp_body = resp.json()
        except Exception:
            resp_body = {"raw": resp.text}

        if resp.status_code >= 400:
            print(f"[api-vapi-call] Vapi error {resp.status_code}: {resp_body}")
            return JSONResponse({"error": "Vapi API error", "details": resp_body}, status_code=resp.status_code)

        print(f"[api-vapi-call] Call initiated: {resp_body.get('id', '?')}")
        return JSONResponse(resp_body)
    except Exception as exc:
        print(f"[api-vapi-call] Request failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=502)


async def api_pairing_revoke(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    platform = body.get("platform", "")
    user_id = body.get("user_id", "")
    if not platform or not user_id:
        return JSONResponse({"error": "platform and user_id required"}, status_code=400)

    approved_path = PAIRING_DIR / f"{platform}-approved.json"
    approved = _load_pairing_json(approved_path)
    if user_id in approved:
        del approved[user_id]
        _save_pairing_json(approved_path, approved)

    return JSONResponse({"ok": True})


async def auto_start_gateway():
    env_vars = read_env_file(ENV_FILE_PATH)
    has_provider = any(env_vars.get(key) for key in PROVIDER_KEYS)
    if has_provider:
        asyncio.create_task(gateway.start())


routes = [
    Route("/", homepage),
    Route("/health", health),
    Route("/api/config", api_config_get, methods=["GET"]),
    Route("/api/config", api_config_put, methods=["PUT"]),
    Route("/api/status", api_status),
    Route("/api/logs", api_logs),
    Route("/api/gateway/start", api_gateway_start, methods=["POST"]),
    Route("/api/gateway/stop", api_gateway_stop, methods=["POST"]),
    Route("/api/gateway/restart", api_gateway_restart, methods=["POST"]),
    Route("/api/pairing/pending", api_pairing_pending),
    Route("/api/pairing/approve", api_pairing_approve, methods=["POST"]),
    Route("/api/pairing/deny", api_pairing_deny, methods=["POST"]),
    Route("/api/pairing/approved", api_pairing_approved),
    Route("/api/pairing/revoke", api_pairing_revoke, methods=["POST"]),
    Route("/api/vapi-call", api_vapi_call, methods=["POST"]),
    Route("/telegram", telegram_webhook_proxy, methods=["POST"]),
    Route("/vapi-webhook", vapi_webhook, methods=["POST"]),
]

@asynccontextmanager
async def lifespan(app):
    await auto_start_gateway()
    yield
    await gateway.stop()


app = Starlette(
    routes=routes,
    middleware=[Middleware(AuthenticationMiddleware, backend=BasicAuthBackend())],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    def handle_signal():
        loop.create_task(gateway.stop())
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    loop.run_until_complete(server.serve())
