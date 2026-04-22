import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

_REQUIRED = ["GREEN_API_INSTANCE_ID", "GREEN_API_TOKEN", "ANTHROPIC_API_KEY"]
_missing = [k for k in _REQUIRED if not os.getenv(k)]
if _missing:
    print(
        f"\n[FATAL] Missing required environment variables: {', '.join(_missing)}\n"
        "The bot cannot start. Set these in your .env file or Render environment settings.\n",
        file=sys.stderr,
    )
    sys.exit(1)

GREEN_API_INSTANCE_ID: str = os.environ["GREEN_API_INSTANCE_ID"]
GREEN_API_TOKEN: str       = os.environ["GREEN_API_TOKEN"]

# URL derived automatically from instance ID (e.g. 7107593885 → https://7107.api.greenapi.com)
_instance_prefix = GREEN_API_INSTANCE_ID[:4] if len(GREEN_API_INSTANCE_ID) >= 4 else GREEN_API_INSTANCE_ID
GREEN_API_URL: str = os.getenv("GREEN_API_URL", f"https://{_instance_prefix}.api.greenapi.com")

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
PORT: int              = int(os.getenv("PORT", "3000"))
TEST_MODE: bool        = os.getenv("TEST_MODE", "false").lower() == "true"

# Normalize TEST_PHONE to WhatsApp chat ID format (e.g. "0529330102" → "9720529330102@c.us")
_raw_phone = os.getenv("TEST_PHONE", "").strip().replace("-", "").replace("+", "")
if _raw_phone.startswith("0"):
    _raw_phone = "972" + _raw_phone[1:]
TEST_PHONE: str = (_raw_phone + "@c.us") if _raw_phone and not _raw_phone.endswith("@c.us") else _raw_phone

GOOGLE_SHEETS_WEBHOOK_URL: str = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL", "")

# Secret token appended as ?token=... in the webhook URL registered with Green-API.
# STRONGLY RECOMMENDED in production — without it, any external party who discovers
# the URL can inject fake messages and burn Claude API budget.
WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")

# Persistent data directory — REQUIRED for production to survive Render restarts.
# Set to a Render Persistent Disk mount path (e.g. /data).
# Defaults to the project root which is EPHEMERAL on Render's service filesystem.
DATA_DIR: str = os.getenv("DATA_DIR", "")

# Admin secret for /diag and /conversations endpoints.
# STRONGLY RECOMMENDED — without it, anyone can read all customer conversation data.
ADMIN_SECRET: str = os.getenv("ADMIN_SECRET", "")

# ── Production safety warnings ────────────────────────────────────────────────
_PROD_WARNINGS: list[str] = []
if not TEST_MODE:
    if not WEBHOOK_SECRET:
        _PROD_WARNINGS.append(
            "WEBHOOK_SECRET is not set — /webhook accepts requests from anyone. "
            "Generate a secret (openssl rand -hex 20), set WEBHOOK_SECRET=<value>, "
            "and update the Green-API webhook URL to include ?token=<value>."
        )
    if not DATA_DIR:
        _PROD_WARNINGS.append(
            "DATA_DIR is not set — leads.json, conversations.json, and dedup_ids.json "
            "are stored in the ephemeral project root and WILL BE WIPED on every Render restart. "
            "Add a Render Persistent Disk, mount it at /data, set DATA_DIR=/data."
        )
    if not ADMIN_SECRET:
        _PROD_WARNINGS.append(
            "ADMIN_SECRET is not set — /diag and /conversations are publicly readable. "
            "Set ADMIN_SECRET=<secret> and access these endpoints with ?admin=<secret>."
        )

# Validate DATA_DIR path is accessible if explicitly set
if DATA_DIR:
    _data_path = Path(DATA_DIR)
    if not _data_path.exists():
        try:
            _data_path.mkdir(parents=True, exist_ok=True)
        except Exception as _e:
            print(
                f"\n[FATAL] DATA_DIR={DATA_DIR!r} does not exist and could not be created: {_e}\n"
                "Fix the DATA_DIR path or remove it to use the default.\n",
                file=sys.stderr,
            )
            sys.exit(1)
    if not os.access(DATA_DIR, os.W_OK):
        print(
            f"\n[FATAL] DATA_DIR={DATA_DIR!r} is not writable. "
            "Check mount permissions on your Render Persistent Disk.\n",
            file=sys.stderr,
        )
        sys.exit(1)
