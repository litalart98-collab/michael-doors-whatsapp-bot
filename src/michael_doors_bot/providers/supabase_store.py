import os
import logging
import httpx

logger = logging.getLogger(__name__)

_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_KEY = os.getenv("SUPABASE_KEY", "")

_HEADERS = {
    "apikey": _KEY,
    "Authorization": f"Bearer {_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}


def _enabled() -> bool:
    return bool(_URL and _KEY)


# ── Conversations ─────────────────────────────────────────────────────────────

async def load_conversation(sender: str) -> list:
    if not _enabled():
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{_URL}/rest/v1/conversations",
                headers=_HEADERS,
                params={"sender": f"eq.{sender}", "select": "messages"},
            )
            r.raise_for_status()
            rows = r.json()
            return rows[0]["messages"] if rows else []
    except Exception as e:
        logger.warning("[SUPABASE] load_conversation failed: %s", e)
        return []


async def save_conversation(sender: str, messages: list) -> None:
    if not _enabled():
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{_URL}/rest/v1/conversations",
                headers=_HEADERS,
                json={"sender": sender, "messages": messages},
            )
    except Exception as e:
        logger.warning("[SUPABASE] save_conversation failed: %s", e)


async def load_all_conversations() -> dict:
    if not _enabled():
        return {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{_URL}/rest/v1/conversations",
                headers=_HEADERS,
                params={"select": "sender,messages"},
            )
            r.raise_for_status()
            return {row["sender"]: row["messages"] for row in r.json()}
    except Exception as e:
        logger.warning("[SUPABASE] load_all_conversations failed: %s", e)
        return {}


async def delete_conversation(sender: str) -> None:
    if not _enabled():
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.delete(
                f"{_URL}/rest/v1/conversations",
                headers=_HEADERS,
                params={"sender": f"eq.{sender}"},
            )
    except Exception as e:
        logger.warning("[SUPABASE] delete_conversation failed: %s", e)


# ── Leads ─────────────────────────────────────────────────────────────────────

async def upsert_lead(lead: dict) -> None:
    if not _enabled():
        return
    try:
        row = {
            "phone":                   lead.get("phone", ""),
            "full_name":               lead.get("full_name"),
            "city":                    lead.get("city"),
            "service_type":            lead.get("service_type"),
            "preferred_contact_hours": lead.get("preferred_contact_hours"),
            "handoff_to_human":        lead.get("handoff_to_human", False),
            "sheets_sent":             lead.get("sheets_sent", False),
            "doors_count":             lead.get("doors_count"),
            "messages":                lead.get("messages", []),
        }
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{_URL}/rest/v1/leads",
                headers=_HEADERS,
                json=row,
            )
    except Exception as e:
        logger.warning("[SUPABASE] upsert_lead failed: %s", e)


# ── Followup ──────────────────────────────────────────────────────────────────

async def load_system_prompt() -> str | None:
    if not _enabled():
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{_URL}/rest/v1/bot_config",
                headers=_HEADERS,
                params={"key": "eq.system_prompt", "select": "value"},
            )
            r.raise_for_status()
            rows = r.json()
            return rows[0]["value"] if rows else None
    except Exception as e:
        logger.warning("[SUPABASE] load_system_prompt failed: %s", e)
        return None


async def load_faq() -> list:
    if not _enabled():
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{_URL}/rest/v1/faq_entries",
                headers=_HEADERS,
                params={"select": "id,category,keywords,answer"},
            )
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning("[SUPABASE] load_faq failed: %s", e)
        return []


async def save_followup(sender: str, state: dict) -> None:
    if not _enabled():
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{_URL}/rest/v1/followup",
                headers=_HEADERS,
                json={"sender": sender, **state},
            )
    except Exception as e:
        logger.warning("[SUPABASE] save_followup failed: %s", e)
