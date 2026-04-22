import logging
import httpx

logger = logging.getLogger(__name__)


async def append_lead(webhook_url: str, row: dict) -> None:
    """POST a lead row to the Google Sheets webhook.
    Raises on failure — callers are responsible for retry/queue logic."""
    if not webhook_url:
        return
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        r = await client.post(webhook_url, json=row)
        r.raise_for_status()
        logger.info("Google Sheets row appended | phone=%s", row.get("phone"))
