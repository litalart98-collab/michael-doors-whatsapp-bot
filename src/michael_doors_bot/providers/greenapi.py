import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class GreenAPIClient:
    def __init__(self, instance_id: str, token: str, api_url: str = "https://api.green-api.com"):
        self._base = f"{api_url.rstrip('/')}/waInstance{instance_id}"
        self._token = token

    async def send_message(self, chat_id: str, message: str) -> dict:
        url = f"{self._base}/sendMessage/{self._token}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):  # 3 attempts: delays 2s, 4s
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.post(url, json={"chatId": chat_id, "message": message})
                    r.raise_for_status()
                    logger.info("Green-API sendMessage OK | chatId=%s | attempt=%d", chat_id, attempt)
                    return r.json()
            except Exception as exc:
                last_exc = exc
                logger.error("Green-API sendMessage failed | attempt=%d/3 | chatId=%s | %s", attempt, chat_id, exc)
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)  # 2s, 4s
        raise last_exc  # type: ignore[misc]

    async def receive_notification(self) -> Optional[dict]:
        """Return next notification body, or None if queue is empty.
        Raises on HTTP error so the poll loop's backoff counter increments."""
        url = f"{self._base}/receiveNotification/{self._token}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            # Green-API returns null JSON when queue is empty
            if not data:
                return None
            return data

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

    async def get_chats(self) -> list[str]:
        """Return list of all existing chat IDs (individual chats only, @c.us suffix)."""
        url = f"{self._base}/getChats/{self._token}"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list):
                    return []
                return [
                    c.get("id") or c.get("chatId", "")
                    for c in data
                    if (c.get("id") or c.get("chatId", "")).endswith("@c.us")
                ]
        except Exception as exc:
            logger.warning("Green-API getChats failed: %s", exc)
            return []

    async def enable_outgoing_webhook(self) -> bool:
        """
        Enable outgoingMessageReceived webhook in Green API settings.
        This is required for the human-takeover feature: when the business owner
        manually sends a message from WhatsApp Web / connected device, Green API
        fires outgoingMessageReceived so the bot knows to step back.

        Green API setSettings docs:
          outgoingWebhook            — "yes"/"no" for manually sent messages
          outgoingAPIMessageWebhook  — "yes"/"no" for API-sent messages (bot)
        We enable the first and disable the second to avoid the bot triggering
        its own takeover when it sends replies.

        Called once at startup.  Safe to call repeatedly — idempotent.
        """
        url = f"{self._base}/setSettings/{self._token}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(url, json={
                    "outgoingWebhook": "yes",
                    "outgoingAPIMessageWebhook": "no",
                })
                data = r.json() if r.content else {}
                if r.status_code == 200 and data.get("saveSettings"):
                    logger.info("[GREEN-API] outgoingWebhook=yes saved — human-takeover ready")
                    return True
                else:
                    logger.warning(
                        "[GREEN-API] setSettings response unexpected: status=%d body=%s",
                        r.status_code, str(data)[:200],
                    )
                    return False
        except Exception as exc:
            logger.warning("[GREEN-API] enable_outgoing_webhook failed: %s", exc)
            return False

    async def get_settings(self) -> dict:
        """Return current Green API instance settings (for diagnostics)."""
        url = f"{self._base}/getSettings/{self._token}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            logger.warning("[GREEN-API] getSettings failed: %s", exc)
            return {}

    async def get_chat_history(self, chat_id: str, count: int = 15) -> list[dict]:
        """Return recent messages for a chat. Each message has:
          type: 'outgoing'/'incoming'
          sendByApi: bool (False = manual owner send, True = API/bot send)
          timestamp: unix int
        Returns empty list on failure (non-fatal).
        """
        url = f"{self._base}/getChatHistory/{self._token}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json={"chatId": chat_id, "count": count})
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("Green-API getChatHistory failed | chatId=%s | %s", chat_id, exc)
            return []

    async def get_contact_name(self, chat_id: str) -> str:
        """Return the WhatsApp display name for a contact, or empty string on failure."""
        url = f"{self._base}/getContactInfo/{self._token}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json={"chatId": chat_id})
                r.raise_for_status()
                data = r.json()
                # Green API returns: name / pushname / contactName (in priority order)
                name = (
                    data.get("name") or
                    data.get("pushname") or
                    data.get("contactName") or
                    ""
                )
                return name.strip()
        except Exception as exc:
            logger.warning("Green-API getContactInfo failed | chatId=%s | %s", chat_id, exc)
            return ""
