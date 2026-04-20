import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

GREEN_API_INSTANCE_ID: str = os.environ["GREEN_API_INSTANCE_ID"]
GREEN_API_TOKEN: str       = os.environ["GREEN_API_TOKEN"]
ANTHROPIC_API_KEY: str     = os.environ["ANTHROPIC_API_KEY"]
PORT: int                  = int(os.getenv("PORT", "3000"))
TEST_MODE: bool            = os.getenv("TEST_MODE", "false").lower() == "true"
TEST_PHONE: str            = os.getenv("TEST_PHONE", "").strip()
