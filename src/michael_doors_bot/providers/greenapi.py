import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)


class GreenAPIClient:
    def __init__(self, instance_id: str, token: str, api_url: str = "https://api.green-api.com"):
        self._base = f"{api_url.rstrip('/')}/waInstance{instance_id}"
        self._token = token

    async def send_message(self, chat_id: str, message: str) -> dict:
        url = f"{self._base}/sendMessage/{self._token}"
        for attempt in range(1, 3):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.post(url, json={"chatId": chat_id, "message": message})
                    r.raise_for_status()
                    logger.info("Green-API sendMessage OK | chatId=%s | attempt=%d", chat_id, attempt)
                    return r.json()
            except Exception as exc:
                logger.error("Green-API sendMessage failed | attempt=%d | chatId=%s | %s", attempt, chat_id, exc)
                if attempt == 2:
                    raise
                await asyncio.sleep(2)

    async def receive_notification(self) -> dict | None:
        url = f"{self._base}/receiveNotification/{self._token}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.error("Green-API receiveNotification failed | %s", exc)
            return None

    async def delete_notification(self, receipt_id: int) -> bool:
        url = f"{self._base}/deleteNotification/{self._token}/{receipt_id}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.delete(url)
                r.raise_for_status()
                return True
        except Exception as exc:
            logger.error("Green-API deleteNotification failed | receiptId=%d | %s", receipt_id, exc)
            return False
