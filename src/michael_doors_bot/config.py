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
