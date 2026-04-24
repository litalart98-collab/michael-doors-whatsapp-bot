import logging
import httpx

logger = logging.getLogger(__name__)


async def append_lead(webhook_url: str, row: dict) -> None:
    """POST a lead row to the Google Sheets Apps Script webhook.
    Google Apps Script returns 302 after processing — treat it as success.
    Raises on failure — callers are responsible for retry/queue logic."""
    if not webhook_url:
        return
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
        r = await client.post(webhook_url, json=row)
        # 302 = Apps Script processed the request successfully (redirect is just the response body)
        if r.status_code in (200, 201, 302):
            logger.info("Google Sheets row appended | phone=%s | status=%d", row.get("phone"), r.status_code)
            return
        r.raise_for_status()
