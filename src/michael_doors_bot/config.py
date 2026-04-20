import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

GREEN_API_INSTANCE_ID: str = os.environ["GREEN_API_INSTANCE_ID"]
GREEN_API_TOKEN: str       = os.environ["GREEN_API_TOKEN"]
# URL derived automatically from instance ID (e.g. 7107593885 → https://7107.api.greenapi.com)
_instance_prefix           = GREEN_API_INSTANCE_ID[:4]
GREEN_API_URL: str         = os.getenv("GREEN_API_URL", f"https://{_instance_prefix}.api.greenapi.com")
ANTHROPIC_API_KEY: str     = os.environ["ANTHROPIC_API_KEY"]
PORT: int                  = int(os.getenv("PORT", "3000"))
TEST_MODE: bool            = os.getenv("TEST_MODE", "false").lower() == "true"
TEST_PHONE: str            = os.getenv("TEST_PHONE", "").strip()
