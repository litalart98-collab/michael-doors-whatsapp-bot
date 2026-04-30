"""
FastAPI server — webhook mode (Green API sends POST on each incoming message).
Polling loop runs as fallback if webhook is not configured.
"""
import asyncio
import json
import logging
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config
from .engine.simple_router import (
    DIAG_STATE as _ROUTER_DIAG,
    ERROR_REPLIES,
    _conversations as _conv_history,
    _conv_state as _router_conv_state,   # per-sender conversation state dict
    clear_conversation,
    generate_conversation_summary,
    get_closing_message,
    get_followup_message,
    get_reply,
    is_closing_intent,
    _is_deferral_intent,
    _is_already_handled_intent,
    _normalize_callback_time,
    _refresh_system_prompt,
    _refresh_faq,
    _save_conv_state,
)
from .engine.messages import (
    FINAL_HANDOFF,
    FINAL_HANDOFF_FEMALE,
    FINAL_HANDOFF_MALE,
    FINAL_HANDOFF_SERVICE,
    FINAL_HANDOFF_SERVICE_FEMALE,
    FINAL_HANDOFF_SERVICE_MALE,
    FINAL_HANDOFF_SHOWROOM,
    FINAL_HANDOFF_SHOWROOM_FEMALE,
    FINAL_HANDOFF_SHOWROOM_MALE,
)
from .providers.greenapi import GreenAPIClient
from .providers.google_sheets import append_lead
from .providers.supabase_store import upsert_lead, save_followup, load_all_conversations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Israel timezone helper ────────────────────────────────────────────────────
# Israel is UTC+2 (winter) / UTC+3 (summer, DST).
# We use the simple approach: try zoneinfo (Python 3.9+), fall back to a
# fixed UTC+3 offset so Render never crashes even without tzdata installed.
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _TZ_IL = _ZoneInfo("Asia/Jerusalem")
except Exception:
    _TZ_IL = None  # type: ignore[assignment]

def _utc_iso_to_il(utc_iso: str) -> str:
    """Convert a UTC ISO timestamp string to Israel local time (DD/MM/YYYY HH:MM)."""
    if not utc_iso:
        return utc_iso
    try:
        dt_utc = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        if _TZ_IL:
            dt_il = dt_utc.astimezone(_TZ_IL)
        else:
            dt_il = dt_utc.astimezone(timezone(timedelta(hours=3)))  # fallback UTC+3
        return dt_il.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return utc_iso


_ROOT     = Path(__file__).parent.parent.parent
# DATA_DIR can be set to a Render Persistent Disk mount (e.g. /data) so runtime
# data files survive service restarts.  Falls back to project root.
_DATA_DIR        = Path(config.DATA_DIR) if config.DATA_DIR else _ROOT
_LEADS_FILE           = _DATA_DIR / "leads.json"
_TEST_LEADS_FILE      = _DATA_DIR / "leads_test.json"
_SESSIONS_FILE        = _DATA_DIR / "sessions.json"
_DEDUP_FILE           = _DATA_DIR / "dedup_ids.json"         # persisted dedup cache
_FOLLOWUP_FILE        = _DATA_DIR / "followup_state.json"    # persisted follow-up timers
_PRE_EXISTING_FILE    = _DATA_DIR / "pre_existing_contacts.json"  # human chats before bot

SESSION_TIMEOUT       = 30 * 60  # seconds
FOLLOWUP_DELAY        = 30 * 60  # 30 min silence → send follow-up
CLOSE_AFTER_FOLLOWUP  = 90 * 60  # 90 min after follow-up → close inquiry (2 h total)

_BOT_ERROR_MSG    = "רגע, בודקת 😊 תכתבו לי שוב בעוד רגע ואענה לכם"
_CONTACT_FALLBACK = "תודה, קיבלנו את ההודעה שלכם. ניצור איתכם קשר בהקדם להמשך טיפול."

# Matches strings that contain ONLY emoji characters (and whitespace).
# Covers: basic emoji, flags, skin-tone/gender modifiers, ZWJ sequences.
_EMOJI_ONLY_RE = re.compile(
    r'^[\s'
    r'\U0001F300-\U0001F9FF'   # Misc symbols, emoticons, transport, etc.
    r'\U00002600-\U000027BF'   # Misc symbols, dingbats
    r'\U0001FA00-\U0001FAFF'   # Extended pictographic
    r'\U00002702-\U000027B0'
    r'\U0000200D'              # ZWJ
    r'\U0000FE0F'              # variation selector
    r'\U00002194-\U00002199'
    r'\U00002B05-\U00002B07'
    r'\U0001F1E0-\U0001F1FF'   # flags
    r'\U0000231A-\U0000231B'
    r'\U000023E9-\U000023F3'
    r'\U000025AA-\U000025AB'
    r'\U000025B6\U000025C0'
    r'\U000025FB-\U000025FE'
    r'\U00002614-\U00002615'
    r'\U00002648-\U00002653'
    r'\U0000267F\U00002693'
    r'\U000026A1\U000026AA-\U000026AB'
    r'\U000026BD-\U000026BE'
    r'\U000026C4-\U000026C5'
    r'\U000026CE\U000026D4'
    r'\U000026EA\U000026F2-\U000026F3'
    r'\U000026F5\U000026FA\U000026FD'
    r'\U00002702\U00002705'
    r'\U00002708-\U0000270D'
    r'\U0000270F\U00002712\U00002714'
    r'\U00002716\U0000271D\U00002721'
    r'\U00002728\U00002733-\U00002734'
    r'\U00002744\U00002747\U0000274C'
    r'\U0000274E\U00002753-\U00002755'
    r'\U00002757\U00002763-\U00002764'
    r'\U00002795-\U00002797\U000027A1'
    r'\U000027B0\U000027BF'
    r'\U00002934-\U00002935'
    r'\U00003030\U0000303D'
    r'\U00003297\U00003299'
    r'\U0001F004\U0001F0CF'
    r']+$'
)

def _is_emoji_only(text: str) -> bool:
    """Return True if the message contains nothing but emoji (and whitespace)."""
    return bool(_EMOJI_ONLY_RE.match(text)) and len(text.strip()) > 0


async def _handle_non_text(sender: str) -> None:
    """
    Handle image / voice / sticker from a customer.

    First non-text:
      Send catalog link and ask for the model name.
      Mark sender in _image_catalog_sent.

    Second non-text (no text reply in between):
      Customer hasn't given a model name → escalate to contact collection.
      Clear the mark so a third image would restart the cycle.
    """
    _followup_reset(sender)
    if sender in _image_catalog_sent:
        # Second image — give up on model name, move to contact collection
        _image_catalog_sent.discard(sender)
        msg = (
            "נראה שקשה לתאר את הדגם בטקסט 😊\n"
            "נציג שלנו ישמח לעזור אישית — אשמח לשם, עיר ומספר טלפון כדי שיחזרו אליכם."
        )
        logger.info("[BOT:IMAGE_ESCALATE] Moving to contact collection | sender=%s", sender)
    else:
        # First image — send catalog link
        _image_catalog_sent.add(sender)
        msg = _build_image_reply(sender)
        logger.info("[BOT:IMAGE_CATALOG] Sent catalog link | sender=%s", sender)
    try:
        await green.send_message(sender, msg)
        _followup_mark_bot_replied(sender)
    except Exception as exc:
        _record_error("send_fail", sender, str(exc))
        logger.error("[BOT:SEND_FAIL] Non-text reply | sender=%s | %s", sender, exc)


def _build_image_reply(sender: str) -> str:
    """Return the non-text reply for a given sender.
    Sends the relevant catalog link(s) based on active topics in the conversation.
    """
    topics = set(_router_conv_state.get(sender, {}).get("active_topics") or [])
    has_entrance = "entrance_doors" in topics
    has_interior = "interior_doors" in topics

    entrance = "https://www.michaeldoors.co.il/catalog/entry-designed"
    interior = "https://www.michaeldoors.co.il/catalog/interior-smooth"

    if has_entrance and not has_interior:
        links = entrance
    elif has_interior and not has_entrance:
        links = interior
    else:
        links = f"{entrance} / {interior}"

    return (
        f"קיבלתי את התמונה 😊 כדי שנוכל להעביר לנציג את הפרטים המדויקים — "
        f"תסתכלו בקטלוג שלנו וציינו את שם הדגם:\n{links}"
    )

# All possible farewell texts — if the last bot message matches any of these,
# the conversation is closed and no follow-up reminder should be sent.
_FAREWELL_TEXTS: frozenset[str] = frozenset({
    FINAL_HANDOFF,
    FINAL_HANDOFF_FEMALE,
    FINAL_HANDOFF_MALE,
    FINAL_HANDOFF_SERVICE,
    FINAL_HANDOFF_SERVICE_FEMALE,
    FINAL_HANDOFF_SERVICE_MALE,
    FINAL_HANDOFF_SHOWROOM,
    FINAL_HANDOFF_SHOWROOM_FEMALE,
    FINAL_HANDOFF_SHOWROOM_MALE,
})

# Max text length passed into _process_message — truncated if exceeded
_MAX_MSG_CHARS = 2000

# ── Message debounce / batching ───────────────────────────────────────────────
# When a customer sends 2–3 messages in quick succession, we wait DEBOUNCE_WINDOW
# seconds after the LAST message before processing anything.  All buffered texts
# are joined (newline-separated) into a single logical input, so the bot replies
# once with full context instead of fragmented partial answers.
DEBOUNCE_WINDOW: float = 3.0   # seconds to wait after the last message

_pending_messages: dict[str, list[str]] = {}   # sender → buffered texts
_debounce_tasks:   dict[str, asyncio.Task] = {} # sender → active sleep task

# ── Diagnostics — error tracking ──────────────────────────────────────────────
_recent_errors: deque = deque(maxlen=50)  # last 50 error events
_error_counts: dict[str, int] = {
    "claude_api":  0,
    "parse":       0,
    "send_fail":   0,
    "webhook":     0,
    "followup":    0,
    "unhandled":   0,
}


def _record_error(kind: str, sender: str, detail: str) -> None:
    _error_counts[kind] = _error_counts.get(kind, 0) + 1
    _recent_errors.append({
        "ts":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "type":   kind,
        "sender": sender[-8:] if sender else "",  # last 8 chars only (privacy)
        "detail": str(detail)[:200],
    })


# ── Image catalog escalation tracker ─────────────────────────────────────────
# Tracks senders who already received the catalog link after sending an image.
# If they send ANOTHER image without giving a model name, we skip to contact
# collection. Cleared when the sender moves to contact collection or closes.
_image_catalog_sent: set[str] = set()

# ── Pre-boot sender block ─────────────────────────────────────────────────────
# Senders whose messages pre-date this bot session (history from before the bot
# was connected to the number).  Once a sender is identified as pre-boot — via
# timestamp comparison — ALL their messages are silently ignored for the
# lifetime of this session.
_pre_boot_senders: set[str] = set()

# ── Pre-existing contacts (persistent) ───────────────────────────────────────
# WhatsApp chats that existed BEFORE the bot was ever connected to this number.
# Populated once (first boot) via Green API getChats, then persisted to disk.
# Senders in this set had human conversations the bot never participated in —
# the bot will silently skip their messages so it never interrupts a pre-bot
# human relationship.  Only contacts with no bot-conversation history are added;
# customers who already talked to the bot are never blocked.
_pre_existing_contacts: set[str] = set()
try:
    _pre_existing_contacts = set(
        json.loads(_PRE_EXISTING_FILE.read_text(encoding="utf-8"))
    )
    logger.info("[BOOT] Pre-existing contacts loaded: %d", len(_pre_existing_contacts))
except FileNotFoundError:
    pass  # first boot — will be populated in _lifespan
except Exception as e:
    logger.warning("[BOOT] Could not load pre-existing contacts: %s", e)


# ── Message deduplication ─────────────────────────────────────────────────────
# Ordered deque so we evict the OLDEST IDs (not a random half) when full.
_MAX_PROCESSED_IDS  = 500
_processed_ids_set: set[str]   = set()
_processed_ids_order: deque    = deque()


def _load_dedup_cache() -> None:
    """Load persisted dedup IDs from disk on startup (Fix 3)."""
    try:
        ids = json.loads(_DEDUP_FILE.read_text(encoding="utf-8"))
        for msg_id in ids[-_MAX_PROCESSED_IDS:]:
            _processed_ids_set.add(msg_id)
            _processed_ids_order.append(msg_id)
        logger.info("[BOOT] Dedup cache loaded: %d IDs from disk", len(_processed_ids_set))
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[BOOT] Dedup cache load failed (starting empty): %s", exc)


def _save_dedup_cache() -> None:
    try:
        _DEDUP_FILE.write_text(
            json.dumps(list(_processed_ids_order)),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[DEDUP] Cache save failed: %s", exc)


_dedup_dirty_count = 0  # batch writes every 10 new IDs


def _is_duplicate(msg_id: str) -> bool:
    return msg_id in _processed_ids_set


def _track_msg_id(msg_id: str) -> None:
    global _dedup_dirty_count
    if not msg_id or msg_id in _processed_ids_set:
        return
    _processed_ids_set.add(msg_id)
    _processed_ids_order.append(msg_id)
    while len(_processed_ids_order) > _MAX_PROCESSED_IDS:
        old = _processed_ids_order.popleft()
        _processed_ids_set.discard(old)
    _dedup_dirty_count += 1
    if _dedup_dirty_count >= 10:
        _save_dedup_cache()
        _dedup_dirty_count = 0


# ── Per-sender rate limiting ──────────────────────────────────────────────────
# Prevents a single sender from flooding the bot and burning Claude API budget.
_RATE_WINDOW  = 60    # seconds
_RATE_MAX_MSG = 10    # max messages per sender per window
_sender_msg_times: dict[str, deque] = {}


def _is_rate_limited(sender: str) -> bool:
    now = time.time()
    times = _sender_msg_times.setdefault(sender, deque())
    while times and now - times[0] > _RATE_WINDOW:
        times.popleft()
    if len(times) >= _RATE_MAX_MSG:
        return True
    times.append(now)
    return False


green = GreenAPIClient(config.GREEN_API_INSTANCE_ID, config.GREEN_API_TOKEN, config.GREEN_API_URL)

# ── Outbound send retry queue (Fix 4) ─────────────────────────────────────────
# When green.send_message exhausts its 3 internal attempts, the message is queued
# here for 2 additional retries (30s apart) before being permanently abandoned.
_SEND_RETRY_DELAY   = 30   # seconds between retries
_SEND_MAX_RETRIES   = 2    # attempts in this queue (on top of greenapi's own 3)
_failed_sends: list[dict] = []   # {"chat_id", "message", "attempts", "next_retry"}


def _queue_send_retry(chat_id: str, message: str) -> None:
    _failed_sends.append({
        "chat_id": chat_id, "message": message,
        "attempts": 1, "next_retry": time.time() + _SEND_RETRY_DELAY,
    })
    # Push the follow-up timer forward so it doesn't fire before the retry resolves.
    # Without this, the follow-up loop could fire 15 min after the customer messaged,
    # racing with the pending retry and confusing the customer.
    if chat_id in _followup and not _followup[chat_id].get("closed"):
        _followup[chat_id]["last_bot_time"] = time.time()
    logger.warning("[BOT:SEND_QUEUED] Message queued for retry in %ds | chat_id=%s", _SEND_RETRY_DELAY, chat_id)


# ── Google Sheets retry queue (Fix 3) ─────────────────────────────────────────
# append_lead now raises on failure. Failed sends are queued here and retried
# up to 3 times with 5-minute delays before being permanently abandoned.
_SHEETS_RETRY_DELAY  = 300  # 5 minutes between retries
_SHEETS_MAX_ATTEMPTS = 3
_sheets_retry_queue: list[dict] = []  # {"row", "is_test", "sender", "attempts", "next_retry"}


def _queue_sheets_retry(row: dict, is_test: bool, sender: str, exc: Exception) -> None:
    _sheets_retry_queue.append({
        "row": row, "is_test": is_test, "sender": sender,
        "attempts": 1, "next_retry": time.time() + _SHEETS_RETRY_DELAY,
    })
    logger.warning("[SHEETS:QUEUED] Sheets send failed — queued for retry | phone=%s | %s", row.get("phone"), exc)


# ── Leads helpers ─────────────────────────────────────────────────────────────
def _leads_path(is_test: bool) -> Path:
    return _TEST_LEADS_FILE if is_test else _LEADS_FILE


def _load_leads(is_test: bool) -> dict:
    f = _leads_path(is_test)
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except Exception:
        return {}


def _save_leads(leads: dict, is_test: bool) -> None:
    try:
        _leads_path(is_test).write_text(json.dumps(leads, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save leads: %s", e)


async def _attach_summary(sender: str, close_reason: str, is_test: bool) -> None:
    """Generate a summary and attach it to the lead record. Runs at most once per conversation."""
    if sender in _summary_attached:
        logger.info("Summary already attached, skipping | sender=%s", sender)
        return
    _summary_attached.add(sender)
    try:
        summary = await generate_conversation_summary(sender, config.ANTHROPIC_API_KEY)
        leads = _load_leads(is_test)
        if sender not in leads:
            # Customer closed before lead was created (e.g. said goodbye on first message)
            leads[sender] = {
                "phone": sender,
                "firstContact": datetime.utcnow().isoformat(),
                "messages": [],
            }
            logger.info("Created minimal lead entry for summary | sender=%s", sender)
        leads[sender]["conv_summary"] = summary
        leads[sender]["close_reason"] = close_reason
        leads[sender]["close_time"] = datetime.utcnow().isoformat()
        _save_leads(leads, is_test)
        logger.info("Summary saved | sender=%s | reason=%s", sender, close_reason)
        # Clear history so a future re-contact starts fresh
        clear_conversation(sender)
    except Exception as exc:
        logger.error("_attach_summary error | sender=%s | %s", sender, exc)
        _summary_attached.discard(sender)  # allow retry on error


def _record_lead(sender: str, user_msg: str, result: dict, is_test: bool) -> dict:
    leads = _load_leads(is_test)
    if sender not in leads:
        leads[sender] = {
            "phone": sender,
            "firstContact": datetime.utcnow().isoformat(),
            "isTest": is_test or None,
            "messages": [],
        }
        logger.info("%sNew lead: %s", "TEST " if is_test else "", sender)

    lead = leads[sender]
    lead["lastMessage"] = datetime.utcnow().isoformat()

    if result.get("preferred_contact_hours"):
        lead["preferred_contact_hours"] = result["preferred_contact_hours"]
    if result.get("needs_frame_removal") is not None:
        lead["needs_frame_removal"] = result["needs_frame_removal"]
    if result.get("needs_installation") is not None:
        lead["needs_installation"] = result["needs_installation"]
    if result.get("full_name"):
        lead["full_name"] = result["full_name"]
    if result.get("phone"):
        lead["callback_phone"] = result["phone"]
    if result.get("service_type"):
        lead["service_type"] = result["service_type"]
    if result.get("city"):
        lead["city"] = result["city"]

    # Record the exact moment all 3 core contact fields are first complete.
    # Used as the Sheets datetime so the timestamp reflects when details were given.
    if (not lead.get("contact_collected_at")
            and lead.get("full_name")
            and lead.get("callback_phone")
            and lead.get("city")):
        lead["contact_collected_at"] = datetime.utcnow().isoformat()
    if result.get("design_preference"):
        lead["design_preference"] = result["design_preference"]
    if result.get("doors_count"):
        lead["doors_count"] = result["doors_count"]
    if result.get("project_status"):
        lead["project_status"] = result["project_status"]
    if result.get("referral_source"):
        lead["referral_source"] = result["referral_source"]
    if result.get("is_returning_customer") is not None:
        lead["is_returning_customer"] = result["is_returning_customer"]
    if result.get("handoff_to_human"):
        lead["handoff_to_human"] = True
        lead["handoff_time"] = datetime.utcnow().isoformat()
    if result.get("summary"):
        lead["summary"] = result["summary"]
    # v2 door detail fields (richer Sheets payload)
    if result.get("active_topics"):
        lead["active_topics"] = result["active_topics"]
    if result.get("entrance_scope"):
        lead["entrance_scope"] = result["entrance_scope"]
    if result.get("entrance_style"):
        lead["entrance_style"] = result["entrance_style"]
    if result.get("entrance_model"):
        lead["entrance_model"] = result["entrance_model"]
    if result.get("interior_project_type"):
        lead["interior_project_type"] = result["interior_project_type"]
    if result.get("interior_quantity"):
        lead["interior_quantity"] = result["interior_quantity"]
    if result.get("interior_style"):
        lead["interior_style"] = result["interior_style"]
    if result.get("interior_model"):
        lead["interior_model"] = result["interior_model"]
    if result.get("mamad_type"):
        lead["mamad_type"] = result["mamad_type"]

    lead["messages"].append({"from": "customer", "text": user_msg, "time": datetime.utcnow().isoformat()})
    if result.get("reply_text"):
        lead["messages"].append({"from": "bot", "text": result["reply_text"], "time": datetime.utcnow().isoformat()})

    _save_leads(leads, is_test)
    return lead


async def _maybe_send_to_sheets(lead: dict, result: dict, is_test: bool) -> None:
    """Send lead to Google Sheets exactly ONCE per conversation, at Stage 7 handoff.

    Rules:
    • Do NOT send partial leads (all 5 required fields must be present).
    • Do NOT send before Stage 7 handoff (handoff_to_human must be True).
    • Once sheets_sent = True → never call append_lead again for this conversation.
      No update rows, no correction rows — one completed conversation = one row.
    """
    if not config.GOOGLE_SHEETS_WEBHOOK_URL:
        return

    # ── One-row guard: already sent → hard stop ───────────────────────────────
    if lead.get("sheets_sent"):
        logger.debug(
            "[SHEETS:SKIP] already sent — one row per conversation | sender=%s",
            lead.get("phone", ""),
        )
        return

    # ── Required fields: 3 core contact fields must be present ──────────────
    # Send to Sheets as soon as name + phone + city are collected — don't wait
    # for preferred_contact_hours.  This ensures every lead that gives contact
    # details reaches Sheets even if the conversation ends early.
    has_core_contact = all(lead.get(f) for f in ("full_name", "callback_phone", "city"))
    if not has_core_contact:
        logger.debug(
            "[SHEETS:SKIP] missing core contact fields | sender=%s | "
            "full_name=%r city=%r callback_phone=%r",
            lead.get("phone", ""), lead.get("full_name"), lead.get("city"),
            lead.get("callback_phone"),
        )
        return

    # Prefer callback phone given by customer; fall back to WhatsApp sender number
    raw_phone = lead.get("callback_phone") or lead.get("phone", "")
    # Normalize to Israeli format: "972529330102@c.us" → "052-9330102"
    phone_clean = raw_phone.replace("@c.us", "").strip()
    if phone_clean.startswith("972") and len(phone_clean) >= 11:
        phone_clean = "0" + phone_clean[3:]
    # Strip dashes/spaces before digit check (Claude may return "052-1234567")
    phone_digits = phone_clean.replace("-", "").replace(" ", "")
    if len(phone_digits) == 10 and phone_digits.isdigit():
        phone_clean = phone_digits[:3] + "-" + phone_digits[3:]

    # ── Build service_field from v2 door details ───────────────────────────────
    active_topics = lead.get("active_topics") or []
    parts_svc: list[str] = []

    if "entrance_doors" in active_topics:
        scope  = lead.get("entrance_scope")
        style  = lead.get("entrance_style")
        model  = lead.get("entrance_model")
        lbl    = "דלת כניסה"
        if style == "flat":
            lbl += " חלקה"
        elif style == "designed":
            lbl += " מעוצבת"
        if scope == "with_frame":
            lbl += " כולל משקוף"
        elif scope == "door_only":
            lbl += " דלת בלבד"
        if model and model not in ("undecided", "לא סוכם"):
            lbl += f" — {model}"
        parts_svc.append(lbl)

    if "interior_doors" in active_topics:
        qty     = lead.get("interior_quantity") or lead.get("doors_count")
        style   = lead.get("interior_style")
        project = lead.get("interior_project_type") or lead.get("project_status")
        model   = lead.get("interior_model")
        lbl     = "דלתות פנים"
        if qty:
            lbl += f" {qty} יח'"
        if style == "flat":
            lbl += " חלקות"
        elif style == "designed":
            lbl += " מעוצבות"
        if project == "new":
            lbl += " — בית חדש"
        elif project == "renovation":
            lbl += " — שיפוץ"
        elif project == "replacement":
            lbl += " — החלפה"
        if model and model not in ("undecided", "לא סוכם"):
            lbl += f" — {model}"
        parts_svc.append(lbl)

    if "mamad" in active_topics:
        mamad_type = lead.get("mamad_type")
        lbl = 'דלת ממ"ד'
        if mamad_type == "new":
            lbl += " חדשה"
        elif mamad_type == "replacement":
            lbl += " — החלפה"
        parts_svc.append(lbl)

    if "repair" in active_topics:
        parts_svc.append("תיקון")

    if "showroom_meeting" in active_topics:
        parts_svc.append("ביקור אולם תצוגה")

    # Fallback to legacy fields for backward-compatibility
    if not parts_svc:
        svc     = lead.get("service_type", "")
        dp      = lead.get("design_preference", "")
        cnt     = lead.get("doors_count")
        frame   = lead.get("needs_frame_removal")
        pstatus = lead.get("project_status", "")
        if svc:
            parts_svc.append(svc)
        if dp and dp != "לא סוכם":
            parts_svc.append(dp)
        if pstatus:
            parts_svc.append(pstatus)
        if frame is True:
            parts_svc.append("כולל החלפת משקוף")
        elif frame is False:
            parts_svc.append("ללא החלפת משקוף")
        if cnt:
            parts_svc.append(f"{cnt} יחידות")

    service_field = " | ".join(parts_svc) if parts_svc else lead.get("service_type", "")

    referral  = lead.get("referral_source", "")
    returning = lead.get("is_returning_customer")
    notes_parts = []
    if referral:
        notes_parts.append(f"הופנה ע\"י: {referral}")
    if returning:
        notes_parts.append("לקוח חוזר")

    # conversation_summary: prefer the full summary from _attach_summary (conv_summary),
    # fall back to the per-turn "summary" field Claude writes.
    conversation_summary = lead.get("conv_summary") or lead.get("summary", "")

    # Normalise callback time to HH:MM before writing to Sheets.
    callback_hours_raw = lead.get("preferred_contact_hours", "")
    callback_hours = _normalize_callback_time(callback_hours_raw)
    if callback_hours != callback_hours_raw:
        logger.debug(
            "[SHEETS:TIME_NORM] %r → %r | sender=%s",
            callback_hours_raw, callback_hours, lead.get("phone", ""),
        )

    row = {
        "full_name":               lead.get("full_name", ""),
        "city":                    lead.get("city", ""),
        "service_type":            service_field,
        "datetime":                _utc_iso_to_il(
            lead.get("contact_collected_at")   # exact moment contact details were given
            or lead.get("firstContact")        # fallback: start of conversation
            or datetime.utcnow().isoformat()   # last-resort fallback
        ),
        "preferred_contact_hours": callback_hours,
        "phone":                   phone_clean,
        "notes":                   " | ".join(notes_parts),
        "conversation_summary":    conversation_summary,
    }

    # Pre-send audit log.
    logger.info(
        "[SHEETS:PRE-SEND] sender=%s | full_name=%r | phone=%s | "
        "service=%r | callback=%r | handoff=%s",
        lead.get("phone", ""), row["full_name"], row["phone"],
        row["service_type"], row["preferred_contact_hours"],
        result.get("handoff_to_human"),
    )

    sender_id = lead.get("phone", "")
    try:
        await append_lead(config.GOOGLE_SHEETS_WEBHOOK_URL, row)
        lead["sheets_sent"] = True
        if sender_id:
            fresh = _load_leads(is_test)
            if sender_id in fresh:
                fresh[sender_id]["sheets_sent"] = True
                _save_leads(fresh, is_test)
        logger.info("[SHEETS:APPENDED] phone=%s | one row written — will not send again", row.get("phone"))
    except Exception as exc:
        _record_error("send_fail", sender_id, f"sheets: {exc}")
        _queue_sheets_retry(row, is_test, sender_id, exc)


async def _send_incomplete_lead_to_sheets(sender: str, is_test: bool) -> None:
    """
    Send a partial lead row to Sheets when a conversation closes without completion.
    Fires only if:
      • Complete lead was NOT already sent (sheets_sent is False)
      • At least one topic was detected
      • Customer sent at least one message (not a silent bounce)
    Row is flagged "❗ דורש מעקב" so the sales team knows to follow up manually.
    """
    if not config.GOOGLE_SHEETS_WEBHOOK_URL:
        return

    # Guard: complete lead already sent — don't add a second row
    leads    = _load_leads(is_test)
    lead_rec = leads.get(sender, {})
    if lead_rec.get("sheets_sent") or lead_rec.get("partial_lead_sent"):
        return

    # Guard: need at least one detected topic
    conv_state = _router_conv_state.get(sender, {})
    active_topics = conv_state.get("active_topics") or []
    if not active_topics:
        return

    # Guard: customer must have sent at least one message
    history = _conv_history.get(sender, [])
    if not any(m.get("role") == "user" for m in history):
        return

    # ── Phone: clean WhatsApp sender ID → Israeli format ─────────────────────
    raw_phone = sender
    phone_clean = raw_phone.replace("@c.us", "").strip()
    if phone_clean.startswith("972") and len(phone_clean) >= 11:
        phone_clean = "0" + phone_clean[3:]
    phone_digits = phone_clean.replace("-", "").replace(" ", "")
    if len(phone_digits) == 10 and phone_digits.isdigit():
        phone_clean = phone_digits[:3] + "-" + phone_digits[3:]

    # ── Service field: same logic as complete leads ───────────────────────────
    parts_svc: list[str] = []
    if "entrance_doors" in active_topics:
        lbl = "דלת כניסה"
        style = conv_state.get("entrance_style")
        if style == "flat":     lbl += " חלקה"
        elif style == "designed": lbl += " מעוצבת"
        elif style == "zero_line": lbl += " קו אפס"
        scope = conv_state.get("entrance_scope")
        if scope == "with_frame": lbl += " כולל משקוף"
        elif scope == "door_only": lbl += " דלת בלבד"
        parts_svc.append(lbl)
    if "interior_doors" in active_topics:
        lbl = "דלתות פנים"
        qty = conv_state.get("interior_quantity")
        if qty: lbl += f" {qty} יח'"
        style = conv_state.get("interior_style")
        if style == "flat":     lbl += " חלקות"
        elif style == "designed": lbl += " מעוצבות"
        parts_svc.append(lbl)
    if "mamad" in active_topics:
        lbl = 'דלת ממ"ד'
        mamad_type = conv_state.get("mamad_type")
        if mamad_type == "new":         lbl += " חדשה"
        elif mamad_type == "replacement": lbl += " — החלפה"
        parts_svc.append(lbl)
    if "repair"           in active_topics: parts_svc.append("תיקון")
    if "showroom_meeting" in active_topics: parts_svc.append("ביקור אולם תצוגה")
    service_field = " | ".join(parts_svc) if parts_svc else ""

    # ── Name: prefer what customer said → fallback to WhatsApp display name ──
    name_from_conv = conv_state.get("full_name") or lead_rec.get("full_name", "")
    if not name_from_conv:
        name_from_conv = await green.get_contact_name(sender)

    row = {
        "full_name":               name_from_conv,
        "city":                    conv_state.get("city") or lead_rec.get("city", ""),
        "service_type":            service_field,
        "datetime":                _utc_iso_to_il(lead_rec.get("firstContact", datetime.utcnow().isoformat())),
        "preferred_contact_hours": "",
        "phone":                   phone_clean,
        "notes":                   "❗ דורש מעקב — לא השלים שיחה",
        "conversation_summary":    lead_rec.get("conv_summary", ""),
    }

    try:
        await append_lead(config.GOOGLE_SHEETS_WEBHOOK_URL, row)
        # Mark so we never send again for this conversation
        leads = _load_leads(is_test)
        if sender not in leads:
            leads[sender] = {"phone": sender}
        leads[sender]["partial_lead_sent"] = True
        _save_leads(leads, is_test)
        logger.info(
            "[SHEETS:INCOMPLETE] Partial lead sent | sender=%s | phone=%s | topics=%s",
            sender, phone_clean, active_topics,
        )
    except Exception as exc:
        logger.error("[SHEETS:INCOMPLETE_ERR] sender=%s | %s", sender, exc)


# ── Follow-up tracker ─────────────────────────────────────────────────────────
# {sender: {"last_bot_time": float, "followup_sent": bool, "followup_time": float, "closed": bool}}
_followup: dict[str, dict] = {}


def _load_followup() -> None:
    """Restore active follow-up timers from disk after a restart."""
    try:
        data = json.loads(_FOLLOWUP_FILE.read_text(encoding="utf-8"))
        cutoff = time.time() - 30 * 60  # discard entries older than 30 min
        restored = 0
        for sender, state in data.items():
            if not state.get("closed") and state.get("last_bot_time", 0) > cutoff:
                _followup[sender] = state
                restored += 1
        logger.info("[BOOT] Follow-up state loaded: %d active entries restored", restored)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[BOOT] Follow-up state load failed (starting empty): %s", exc)


def _save_followup() -> None:
    try:
        _FOLLOWUP_FILE.write_text(json.dumps(_followup), encoding="utf-8")
    except Exception as exc:
        logger.warning("[FOLLOWUP] State save failed: %s", exc)


# Guard: prevent _attach_summary from running twice for the same conversation
_summary_attached: set[str] = set()

_CLOSE_MSG = (
    "ראיתי שלא חזרתם, אז נסגור את הפנייה לעת עתה 😊\n"
    "אם תרצו לחזור בכל שעה ולהתחיל שיחה חדשה — נשמח לעזור!\n"
    "דלתות מיכאל | 054-2787578"
)


def _followup_reset(sender: str) -> None:
    state = _followup.get(sender)
    if state and state.get("closed"):
        _followup[sender] = {"last_bot_time": time.time(), "followup_sent": False, "followup_time": 0.0, "closed": False}
    elif state:
        state["last_bot_time"] = time.time()
        state["followup_sent"] = False
        state["followup_time"] = 0.0
    else:
        _followup[sender] = {"last_bot_time": time.time(), "followup_sent": False, "followup_time": 0.0, "closed": False}
    _save_followup()


def _followup_mark_bot_replied(sender: str) -> None:
    if sender not in _followup:
        _followup[sender] = {"last_bot_time": time.time(), "followup_sent": False, "followup_time": 0.0, "closed": False}
    else:
        _followup[sender]["last_bot_time"] = time.time()
        _followup[sender]["closed"] = False
    _save_followup()


async def _followup_loop() -> None:
    logger.info("Follow-up loop started")
    _tick = 0
    while True:
        await asyncio.sleep(60)
        _tick += 1
        now = time.time()

        # Hourly cleanup: remove closed conversations older than 2 hours to prevent memory leaks
        if _tick % 60 == 0:
            cutoff = now - 2 * 3600
            stale = [
                s for s, st in _followup.items()
                if st.get("closed") and st.get("followup_time", 0) < cutoff
            ]
            for s in stale:
                _followup.pop(s, None)
                _sender_locks.pop(s, None)
                _summary_attached.discard(s)
            if stale:
                logger.info("Memory cleanup: removed %d stale conversations", len(stale))
                _save_followup()

        # ── Drain outbound send retry queue ───────────────────────────────────
        still_pending_sends: list[dict] = []
        for entry in _failed_sends:
            if now < entry["next_retry"]:
                still_pending_sends.append(entry)
                continue
            try:
                await green.send_message(entry["chat_id"], entry["message"])
                logger.info("[BOT:SEND_RETRY_OK] Retry succeeded | chat_id=%s | attempt=%d",
                            entry["chat_id"], entry["attempts"])
            except Exception as exc:
                entry["attempts"] += 1
                if entry["attempts"] > _SEND_MAX_RETRIES:
                    logger.critical("[BOT:SEND_PERMANENT_FAIL] All retry attempts exhausted — "
                                    "message permanently lost | chat_id=%s | text=%s",
                                    entry["chat_id"], entry["message"][:60])
                else:
                    entry["next_retry"] = now + _SEND_RETRY_DELAY
                    still_pending_sends.append(entry)
                    logger.warning("[BOT:SEND_RETRY_FAIL] attempt=%d | chat_id=%s | %s",
                                   entry["attempts"], entry["chat_id"], exc)
        _failed_sends[:] = still_pending_sends

        # ── Drain Google Sheets retry queue ───────────────────────────────────
        still_pending_sheets: list[dict] = []
        for entry in _sheets_retry_queue:
            if now < entry["next_retry"]:
                still_pending_sheets.append(entry)
                continue
            try:
                await append_lead(config.GOOGLE_SHEETS_WEBHOOK_URL, entry["row"])
                fresh = _load_leads(entry["is_test"])
                if entry["sender"] in fresh:
                    fresh[entry["sender"]]["sheets_sent"] = True
                    _save_leads(fresh, entry["is_test"])
                logger.info("[SHEETS:RETRY_OK] Retry succeeded | phone=%s | attempt=%d",
                            entry["row"].get("phone"), entry["attempts"])
            except Exception as exc:
                entry["attempts"] += 1
                if entry["attempts"] >= _SHEETS_MAX_ATTEMPTS:
                    logger.critical("[SHEETS:PERMANENT_FAIL] All %d attempts failed — lead not in Sheets | "
                                    "phone=%s | %s", _SHEETS_MAX_ATTEMPTS, entry["row"].get("phone"), exc)
                else:
                    entry["next_retry"] = now + _SHEETS_RETRY_DELAY
                    still_pending_sheets.append(entry)
                    logger.warning("[SHEETS:RETRY_FAIL] attempt=%d | phone=%s | %s",
                                   entry["attempts"], entry["row"].get("phone"), exc)
        _sheets_retry_queue[:] = still_pending_sheets

        for sender, state in list(_followup.items()):
            if not _is_individual_chat(sender):
                continue
            if state.get("closed"):
                continue
            last_bot = state.get("last_bot_time", 0.0)
            followup_sent = state.get("followup_sent", False)
            followup_time = state.get("followup_time", 0.0)

            if not followup_sent and now - last_bot >= FOLLOWUP_DELAY:
                history = _conv_history.get(sender, [])

                # Guard: never follow-up if no customer reply is recorded in history.
                # This means the bot sent its initial pitch but never received a reply
                # (webhook miss, poll error, etc.). The customer is not "going silent" —
                # they may never have seen our message. Sending a follow-up would be wrong.
                customer_has_replied = any(m.get("role") == "user" for m in history)
                if not customer_has_replied:
                    logger.info("[BOT:FOLLOWUP_SKIP] No customer reply in history | sender=%s", sender)
                    continue

                # Guard: never follow-up while a send-retry is queued for this sender.
                # The bot's reply may still land — firing the follow-up first is confusing.
                if any(e["chat_id"] == sender for e in _failed_sends):
                    logger.info("[BOT:FOLLOWUP_SKIP] Pending send retry — deferring follow-up | sender=%s", sender)
                    continue

                # If the last bot message was an error, stop watching entirely.
                # Setting followup_sent=True here was a bug: it would trigger the
                # 7-min closing sequence, sending a generic close to a customer who
                # got an error and may simply retry. Instead we drop the watch entry
                # so no follow-up and no close fires; the customer can write again fresh.
                last_bot_text = next((m["content"] for m in reversed(history) if m.get("role") == "assistant"), "")
                if last_bot_text in ERROR_REPLIES:
                    logger.info("[BOT:FOLLOWUP_SKIP] Dropping watch after error reply | sender=%s", sender)
                    _followup.pop(sender, None)
                    _save_followup()
                    continue

                # ── Conversation-complete guards ──────────────────────────────
                # Guard 1: router state — preferred_contact_hours or handoff_to_human
                # mean the conversation reached Stage 7 (callback time answered / farewell sent).
                # Either one is sufficient — close silently, no reminder.
                conv_state = _router_conv_state.get(sender, {})
                if (
                    conv_state.get("handoff_to_human")
                    or conv_state.get("preferred_contact_hours")
                ):
                    logger.info(
                        "[BOT:FOLLOWUP_SKIP] Conv complete (handoff=%s callback=%s) — "
                        "closing silently | sender=%s",
                        conv_state.get("handoff_to_human"),
                        bool(conv_state.get("preferred_contact_hours")),
                        sender,
                    )
                    state["closed"] = True
                    _save_followup()
                    continue

                # Guard 2: lead record — already sent to Sheets or handoff recorded.
                lead_rec = _load_leads(config.TEST_MODE).get(sender, {})
                if lead_rec.get("sheets_sent") or lead_rec.get("handoff_to_human"):
                    logger.info(
                        "[BOT:FOLLOWUP_SKIP] Lead already complete "
                        "(sheets_sent=%s handoff=%s) | sender=%s",
                        lead_rec.get("sheets_sent"), lead_rec.get("handoff_to_human"), sender,
                    )
                    state["closed"] = True
                    _save_followup()
                    continue

                # Guard 3: conversation history — last bot message was a farewell.
                if last_bot_text in _FAREWELL_TEXTS:
                    logger.info(
                        "[BOT:FOLLOWUP_SKIP] Farewell already sent in history | sender=%s", sender
                    )
                    state["closed"] = True
                    _save_followup()
                    continue

                # Guard 4: last bot message is not a question — nothing to wait for.
                # If the bot's last reply has no "?" it wasn't waiting for a customer
                # answer (e.g. "אשמח לעזור כשנחזור לעבודה ביום רביעי 😊"), so skip
                # the follow-up entirely and close the watch entry quietly.
                if "?" not in last_bot_text:
                    logger.info(
                        "[BOT:FOLLOWUP_SKIP] Last bot message has no question — nothing to follow up | sender=%s",
                        sender,
                    )
                    state["closed"] = True
                    _save_followup()
                    continue

                try:
                    msg = await get_followup_message(sender, config.ANTHROPIC_API_KEY)
                    await green.send_message(sender, msg)
                    state["followup_sent"] = True
                    state["followup_time"] = time.time()
                    _save_followup()
                    logger.info("[BOT:FOLLOWUP] Sent | sender=%s", sender)
                except Exception as exc:
                    _record_error("followup", sender, str(exc))
                    logger.error("[BOT:FOLLOWUP_ERR] sender=%s | %s", sender, exc)

            elif followup_sent and now - followup_time >= CLOSE_AFTER_FOLLOWUP:
                # Close silently — no message sent to customer.
                # We still close the state internally and save the incomplete lead to Sheets.
                state["closed"] = True
                _save_followup()
                logger.info("[BOT:CLOSE] Inquiry closed silently (no-response) | sender=%s", sender)
                try:
                    await _attach_summary(sender, "נסגרה ללא מענה", config.TEST_MODE)
                    await _send_incomplete_lead_to_sheets(sender, config.TEST_MODE)
                except Exception as exc:
                    logger.error("Close (silent) error | sender=%s | %s", sender, exc)


# ── Session helpers (test mode) ───────────────────────────────────────────────
_sessions: dict[str, float] = {}
try:
    _sessions = json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
except Exception:
    pass


def _save_sessions() -> None:
    try:
        _SESSIONS_FILE.write_text(json.dumps(_sessions), encoding="utf-8")
    except Exception:
        pass


def _has_active_session(sender: str) -> bool:
    ts = _sessions.get(sender)
    if ts is None:
        return False
    if time.time() - ts > SESSION_TIMEOUT:
        del _sessions[sender]
        _save_sessions()
        return False
    return True


def _open_session(sender: str) -> None:
    _sessions[sender] = time.time()
    _save_sessions()
    clear_conversation(sender)
    logger.info("TEST SESSION | Opened | %s", sender)


def _touch_session(sender: str) -> None:
    _sessions[sender] = time.time()
    _save_sessions()


def _close_session(sender: str) -> None:
    _sessions.pop(sender, None)
    _save_sessions()
    logger.info("TEST SESSION | Closed by #endtest | %s", sender)


# ── Message processor ─────────────────────────────────────────────────────────
def _is_individual_chat(sender: str) -> bool:
    return sender.endswith("@c.us")


# Per-sender locks: messages from the same sender are processed one at a time;
# messages from different senders run in parallel without blocking each other.
_sender_locks: dict[str, asyncio.Lock] = {}


def _get_sender_lock(sender: str) -> asyncio.Lock:
    if sender not in _sender_locks:
        _sender_locks[sender] = asyncio.Lock()
    return _sender_locks[sender]


# ── Debounce helpers ──────────────────────────────────────────────────────────

def _schedule_debounced(sender: str, text: str) -> None:
    """Buffer text for sender and (re)start the debounce timer.

    Called from both the webhook handler and the poll loop.  Returns immediately;
    actual processing happens DEBOUNCE_WINDOW seconds after the *last* call for
    this sender, via a background asyncio Task.
    """
    _pending_messages.setdefault(sender, []).append(text)

    # Cancel any outstanding timer — a new message resets the window
    existing = _debounce_tasks.get(sender)
    if existing and not existing.done():
        existing.cancel()

    task = asyncio.create_task(_debounce_wrapper(sender))
    _debounce_tasks[sender] = task
    logger.info(
        "[DEBOUNCE:SCHED] sender=%s | buffer=%d msg(s) | window=%.1fs",
        sender, len(_pending_messages[sender]), DEBOUNCE_WINDOW,
    )


async def _debounce_wrapper(sender: str) -> None:
    """Sleep for DEBOUNCE_WINDOW, then flush all buffered messages."""
    try:
        await asyncio.sleep(DEBOUNCE_WINDOW)
    except asyncio.CancelledError:
        return  # a newer message arrived — the new task will handle flushing
    await _flush_pending(sender)


async def _flush_pending(sender: str) -> None:
    """Combine all buffered messages for sender and run the pipeline once."""
    _debounce_tasks.pop(sender, None)
    parts = _pending_messages.pop(sender, [])
    if not parts:
        return
    combined = "\n".join(parts)
    if len(parts) > 1:
        logger.info(
            "[DEBOUNCE:FLUSH] sender=%s | merged %d messages → %d chars | texts=%s",
            sender, len(parts), len(combined),
            " | ".join(p[:40] for p in parts),
        )
    await _process_message(sender, combined)


async def _process_message(sender: str, text: str) -> None:
    # Rate limiting — checked before acquiring lock so flooded senders don't queue (Fix 6)
    if _is_rate_limited(sender):
        logger.warning("[BOT:RATE_LIMIT] Dropped message | sender=%s | text=%s", sender, text[:40])
        return

    # Truncate extreme-length input before acquiring lock
    if len(text) > _MAX_MSG_CHARS:
        logger.warning("[BOT:TRUNCATE] Input truncated %d→%d | sender=%s", len(text), _MAX_MSG_CHARS, sender)
        text = text[:_MAX_MSG_CHARS]

    async with _get_sender_lock(sender):
        try:
            if not _is_individual_chat(sender):
                logger.info("[BOT:SKIP] Non-individual chat | sender=%s", sender)
                return
            if config.TEST_MODE:
                if sender != config.TEST_PHONE:
                    logger.info("[BOT:BLOCKED] TEST_MODE blocked sender=%s", sender)
                    return
                if text.strip() == "#reset":
                    clear_conversation(sender)
                    _followup.pop(sender, None)
                    _summary_attached.discard(sender)
                    await green.send_message(sender, "שיחה אופסה ✓")
                    return

            logger.info("[BOT:RECV] sender=%s | chars=%d | text=%s", sender, len(text), text[:60])

            # Customer replied — reset follow-up timer
            _followup_reset(sender)

            # Smart closing: if customer says goodbye/thanks mid-conversation, close gracefully
            conv_turns = len(_conv_history.get(sender, []))
            last_bot_msg = ""
            history = _conv_history.get(sender, [])
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_bot_msg = msg.get("content", "")
                    break
            # Fix 7: suppress closing intent if the bot's last message was a question
            # (covers "הכל נכון?", "מה מספרך?", "מתי נוח?", etc.)
            bot_last_is_question = "?" in last_bot_msg
            if is_closing_intent(text, conv_turns) and not bot_last_is_question:
                stripped_text = text.strip()
                if _is_already_handled_intent(stripped_text):
                    close_reason = "handled"
                elif _is_deferral_intent(stripped_text):
                    close_reason = "deferred"
                else:
                    close_reason = "farewell"
                logger.info("[BOT:CLOSE] Closing intent | reason=%s | sender=%s", close_reason, sender)
                closing_msg = await get_closing_message(sender, config.ANTHROPIC_API_KEY, reason=close_reason)
                try:
                    await green.send_message(sender, closing_msg)
                    logger.info("[BOT:SEND] Closing message sent | sender=%s", sender)
                    _followup[sender] = {
                        "last_bot_time": time.time(),
                        "followup_sent": True,
                        "followup_time": time.time(),
                        "closed": True,
                    }
                    _save_followup()
                    await _attach_summary(sender, "סגירה בידידות", config.TEST_MODE)
                except Exception as send_err:
                    _record_error("send_fail", sender, str(send_err))
                    logger.error("[BOT:SEND_FAIL] Closing send failed | sender=%s | %s", sender, send_err)
                    _queue_send_retry(sender, closing_msg)
                return

            result = await get_reply(sender, text, config.ANTHROPIC_API_KEY)
            lead = _record_lead(sender, text, result, config.TEST_MODE)
            await _maybe_send_to_sheets(lead, result, config.TEST_MODE)
            await upsert_lead(lead)
            if result.get("handoff_to_human"):
                await _attach_summary(sender, "הועבר לנציג", config.TEST_MODE)
            reply_text = result["reply_text"]
            reply_text_2 = result.get("reply_text_2")  # second pulse (opening message only)
            is_fallback = reply_text in ERROR_REPLIES
            try:
                await green.send_message(sender, reply_text)
                if is_fallback:
                    _record_error("parse", sender, "fallback reply sent after error")
                    logger.warning("[BOT:FALLBACK] Error fallback sent | sender=%s", sender)
                else:
                    logger.info("[BOT:SEND] Reply sent | sender=%s | chars=%d", sender, len(reply_text))
            except Exception as send_err:
                _record_error("send_fail", sender, str(send_err))
                logger.error("[BOT:SEND_FAIL] Reply delivery failed | sender=%s | %s", sender, send_err)
                if not is_fallback:
                    _queue_send_retry(sender, reply_text)

            # Second pulse (opening message split) — separate try so a failure here
            # never blocks follow-up tracking or retries the first pulse redundantly.
            if reply_text_2 and not is_fallback:
                try:
                    await asyncio.sleep(2.5)  # generous pause to avoid Green API rate limit
                    await green.send_message(sender, reply_text_2)
                    logger.info("[BOT:SEND2] Second pulse sent | sender=%s", sender)
                except Exception as send_err2:
                    _record_error("send_fail", sender, f"pulse2: {send_err2}")
                    logger.error("[BOT:SEND2_FAIL] Second pulse failed | sender=%s | %s", sender, send_err2)
                    _queue_send_retry(sender, reply_text_2)

            # Update follow-up state regardless of whether the second pulse succeeded
            if result.get("handoff_to_human"):
                if sender in _followup:
                    _followup[sender]["closed"] = True
                else:
                    _followup[sender] = {
                        "last_bot_time": time.time(),
                        "followup_sent": True,
                        "followup_time": time.time(),
                        "closed": True,
                    }
                _save_followup()
            else:
                _followup_mark_bot_replied(sender)
        except Exception as exc:
            _record_error("unhandled", sender, str(exc))
            logger.error("[BOT:UNHANDLED] _process_message crash | sender=%s | %s", sender, exc)
            # Clear follow-up watch — the conversation state is unknown after a crash.
            # Without this, a follow-up would fire 15 min later from a broken history state.
            _followup.pop(sender, None)
            _save_followup()
            try:
                if _is_individual_chat(sender):
                    await green.send_message(sender, _BOT_ERROR_MSG)
                    logger.info("[BOT:FALLBACK] Sent fallback after crash | sender=%s", sender)
            except Exception:
                pass


# ── Polling loop (fallback when no webhook configured) ────────────────────────
async def _poll_loop() -> None:
    logger.info("Polling loop started (fallback mode)")
    _consecutive_errors = 0
    while True:
        try:
            notification = await green.receive_notification()
            _consecutive_errors = 0  # reset on success
            if not notification:
                await asyncio.sleep(2)
                continue

            receipt_id = notification.get("receiptId")
            body = notification.get("body", {})

            if body.get("typeWebhook") == "incomingMessageReceived":
                sender   = body.get("senderData", {}).get("chatId", "")
                msg_data = body.get("messageData", {})
                msg_id   = body.get("idMessage", "")
                text = (
                    msg_data.get("textMessageData", {}).get("textMessage")
                    or msg_data.get("extendedTextMessageData", {}).get("text")
                    or ""
                )
                msg_timestamp = body.get("timestamp", 0)
                if msg_timestamp and int(msg_timestamp) < int(_bot_start_time):
                    if sender:
                        _pre_boot_senders.add(sender)
                    logger.info(
                        "[BOT:SKIP_OLD] Poll pre-boot message — sender blocked | id=%s | sender=%s",
                        msg_id, sender,
                    )
                elif sender in _pre_boot_senders:
                    logger.info("[BOT:SKIP_PREBOOT] Poll blocked sender | sender=%s", sender)
                elif sender in _pre_existing_contacts:
                    logger.info("[BOT:SKIP_PREEXISTING] Poll pre-bot human contact | sender=%s", sender)
                elif sender and msg_id and _is_duplicate(msg_id):
                    logger.info("[BOT:DEDUP] Poll duplicate skipped | id=%s | sender=%s", msg_id, sender)
                elif sender and not text and _is_individual_chat(sender):
                    msg_type = msg_data.get("typeMessage", "")
                    if msg_type and msg_type not in ("textMessage", "extendedTextMessage", ""):
                        logger.info("[BOT:NON_TEXT] type=%s | sender=%s", msg_type, sender)
                        await _handle_non_text(sender)
                elif sender and text:
                    _image_catalog_sent.discard(sender)  # text reply clears image escalation
                    _track_msg_id(msg_id)
                    if _is_emoji_only(text):
                        # Emoji-only = acknowledgment ("👌", "😊" etc.) — reset timer, no reply.
                        _followup_reset(sender)
                        logger.info("[BOT:EMOJI_ACK] Emoji-only msg, timer reset | sender=%s", sender)
                    else:
                        logger.info("[BOT:RECV] Poll | sender=%s | text=%s", sender, text[:60])
                        _schedule_debounced(sender, text)

            if receipt_id:
                await green.delete_notification(receipt_id)

        except Exception as exc:
            _consecutive_errors += 1
            backoff = min(5 * (2 ** (_consecutive_errors - 1)), 120)
            _record_error("webhook", "", f"poll_loop error #{_consecutive_errors}: {exc}")
            logger.error("[BOT:POLL_ERR] count=%d %s — retry in %ds", _consecutive_errors, exc, backoff)
            await asyncio.sleep(backoff)


# ── Background task supervisor ────────────────────────────────────────────────
async def _supervised(name: str, coro_fn) -> None:
    """Run an infinite-loop background task, restarting it if it ever crashes."""
    while True:
        try:
            await coro_fn()
        except asyncio.CancelledError:
            logger.info("Background task %s cancelled", name)
            raise
        except Exception as exc:
            logger.critical("Background task %s crashed — restarting in 5s | %s", name, exc)
            await asyncio.sleep(5)


# ── FastAPI app ───────────────────────────────────────────────────────────────
_poll_task: Optional[asyncio.Task] = None
_followup_task: Optional[asyncio.Task] = None
_bot_start_time: float = 0.0


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _poll_task, _followup_task, _bot_start_time
    _bot_start_time = time.time()

    logger.info("=== BOT STARTING ===")
    logger.info("TEST_MODE=%s | TEST_PHONE=%s", config.TEST_MODE, config.TEST_PHONE or "(not set)")
    logger.info("GREEN_API_INSTANCE=%s | URL=%s", config.GREEN_API_INSTANCE_ID, config.GREEN_API_URL)
    logger.info("ANTHROPIC_API_KEY set=%s", bool(config.ANTHROPIC_API_KEY))
    logger.info("GOOGLE_SHEETS configured=%s", bool(config.GOOGLE_SHEETS_WEBHOOK_URL))
    logger.info("WEBHOOK_SECRET set=%s", bool(config.WEBHOOK_SECRET))
    logger.info("DATA_DIR=%s", str(_DATA_DIR))

    for _warn in config._PROD_WARNINGS:
        logger.critical("[BOOT:PROD_WARNING] %s", _warn)

    if config.TEST_MODE and not config.TEST_PHONE:
        logger.warning("TEST_MODE is ON but TEST_PHONE is not set — ALL incoming messages will be blocked!")

    _load_dedup_cache()
    _load_followup()

    # Load system prompt and FAQ from Supabase (overrides file-based fallback)
    await _refresh_system_prompt()
    await _refresh_faq()

    # Load conversation history from Supabase (fills in what's not already in memory)
    supabase_convs = await load_all_conversations()
    for sender, msgs in supabase_convs.items():
        if sender not in _conv_history:
            _conv_history[sender] = msgs
    if supabase_convs:
        logger.info("[BOOT] Supabase conversations loaded: %d senders", len(supabase_convs))

    # ── Pre-existing contacts (first-boot population) ─────────────────────────
    # If the file doesn't exist yet, this is the first time the bot runs on this
    # number.  Fetch all existing WhatsApp chats via Green API and save them as
    # "pre-existing contacts" — conversations the bot never managed.
    # Contacts already in _conv_history (existing bot conversations) are excluded
    # so returning bot customers are never blocked.
    global _pre_existing_contacts
    if not _PRE_EXISTING_FILE.exists():
        logger.info("[BOOT] Pre-existing contacts file not found — fetching from Green API…")
        try:
            all_chats = await green.get_chats()
            bot_known = set(_conv_history.keys())
            new_pre_existing = {c for c in all_chats if c not in bot_known}
            _pre_existing_contacts = new_pre_existing
            _PRE_EXISTING_FILE.write_text(
                json.dumps(sorted(_pre_existing_contacts), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "[BOOT] Pre-existing contacts saved: %d (of %d total chats, %d excluded as bot history)",
                len(_pre_existing_contacts), len(all_chats), len(bot_known),
            )
        except Exception as exc:
            logger.warning("[BOOT] Could not fetch pre-existing contacts: %s — no contacts blocked", exc)

    _poll_task    = asyncio.create_task(_supervised("poll_loop",    _poll_loop))
    _followup_task = asyncio.create_task(_supervised("followup_loop", _followup_loop))
    yield
    _poll_task.cancel()
    _followup_task.cancel()
    logger.info("=== BOT SHUTTING DOWN ===")


app = FastAPI(lifespan=_lifespan)


@app.get("/", response_class=JSONResponse)
async def health():
    uptime_s = int(time.time() - _bot_start_time) if _bot_start_time else 0
    return {
        "status": "ok",
        "uptime_seconds": uptime_s,
        "active_conversations": len(_conv_history),
        "active_followups": sum(1 for s in _followup.values() if not s.get("closed")),
        "poll_task_alive": _poll_task is not None and not _poll_task.done(),
        "followup_task_alive": _followup_task is not None and not _followup_task.done(),
        "test_mode": config.TEST_MODE,
        "admin_secret_set": bool(config.ADMIN_SECRET),
        "admin_secret_len": len(config.ADMIN_SECRET),
    }


@app.post("/webhook")
async def webhook(request: Request, token: str = Query(default="")):
    # Fix 5: reject requests that don't carry the expected secret token (if configured).
    # Register the webhook URL in Green-API as: https://your-app.onrender.com/webhook?token=SECRET
    if config.WEBHOOK_SECRET and token != config.WEBHOOK_SECRET:
        logger.warning("[BOT:AUTH_FAIL] Webhook rejected — bad/missing token from %s",
                       getattr(request.client, "host", "unknown"))
        return JSONResponse({"ok": False}, status_code=403)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    logger.info("Webhook received: typeWebhook=%s", body.get("typeWebhook"))

    if body.get("typeWebhook") != "incomingMessageReceived":
        return JSONResponse({"ok": True})

    sender   = body.get("senderData", {}).get("chatId", "")
    msg_data = body.get("messageData", {})
    msg_id   = body.get("idMessage", "")
    text = (
        msg_data.get("textMessageData", {}).get("textMessage")
        or msg_data.get("extendedTextMessageData", {}).get("text")
        or ""
    )

    # ── Pre-boot sender block ─────────────────────────────────────────────────
    # Layer 1: timestamp filter — message predates bot startup → mark sender
    msg_timestamp = body.get("timestamp", 0)
    if msg_timestamp and int(msg_timestamp) < int(_bot_start_time):
        if sender:
            _pre_boot_senders.add(sender)
        logger.info(
            "[BOT:SKIP_OLD] Pre-boot message — sender blocked | id=%s | sender=%s | "
            "msg_ts=%d | boot_ts=%d",
            msg_id, sender, int(msg_timestamp), int(_bot_start_time),
        )
        return JSONResponse({"ok": True})

    # Layer 2: sender already known to have pre-boot history → block all their messages
    if sender in _pre_boot_senders:
        logger.info("[BOT:SKIP_PREBOOT] Blocked sender has pre-boot history | sender=%s", sender)
        return JSONResponse({"ok": True})

    # Pre-existing contacts — human conversations before bot was connected.
    # Never respond to these; the bot was not part of their conversation history.
    if sender in _pre_existing_contacts:
        logger.info("[BOT:SKIP_PREEXISTING] Pre-bot human contact — staying silent | sender=%s", sender)
        return JSONResponse({"ok": True})

    # Deduplication — webhook and poll loop can both deliver the same message
    if sender and msg_id:
        if _is_duplicate(msg_id):
            logger.info("[BOT:DEDUP] Webhook duplicate skipped | id=%s | sender=%s", msg_id, sender)
            return JSONResponse({"ok": True})
        _track_msg_id(msg_id)

    # Non-text messages (images, voice notes, stickers, etc.)
    if sender and not text and _is_individual_chat(sender):
        msg_type = msg_data.get("typeMessage", "")
        if msg_type and msg_type not in ("textMessage", "extendedTextMessage", ""):
            logger.info("[BOT:NON_TEXT] type=%s | sender=%s", msg_type, sender)
            await _handle_non_text(sender)
        return JSONResponse({"ok": True})

    if sender and text:
        _image_catalog_sent.discard(sender)  # text reply clears image escalation state
        if _is_emoji_only(text):
            _followup_reset(sender)
            logger.info("[BOT:EMOJI_ACK] Emoji-only msg, timer reset | sender=%s", sender)
            return JSONResponse({"ok": True})
        logger.info("[BOT:RECV] Webhook | sender=%s | chars=%d | text=%s", sender, len(text), text[:60])
        # Fire-and-forget: buffer the message and let the debounce timer decide
        # when to process.  Multiple messages sent in quick succession are merged
        # into one pipeline call after DEBOUNCE_WINDOW seconds of silence.
        _schedule_debounced(sender, text)

    return JSONResponse({"ok": True})


@app.get("/debug-test", response_class=JSONResponse)
async def debug_test():
    """Full synchronous test — processes message and returns result."""
    try:
        sender = "972500000000@c.us"
        text = "שלום"
        result = await get_reply(sender, text, config.ANTHROPIC_API_KEY)
        _record_lead(sender, text, result, False)
        leads = _load_leads(False)
        return {
            "ok": True,
            "reply": result["reply_text"],
            "green_api_url": config.GREEN_API_URL,
            "leads_path": str(_LEADS_FILE),
            "leads_count": len(leads),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/test-chat")
async def test_chat(request: Request):
    """Browser test UI endpoint — processes a message without sending to WhatsApp.
    Set mock=true to skip Anthropic API entirely (no credits used)."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    sender  = (data.get("sender") or "test_ui_default@c.us").strip()
    message = (data.get("message") or "").strip()
    reset   = bool(data.get("reset", False))
    mock    = bool(data.get("mock", True))

    if reset:
        clear_conversation(sender)
        return JSONResponse({"ok": True, "reset": True})

    if not message:
        return JSONResponse({"ok": False, "error": "empty message"}, status_code=400)

    api_key = config.ANTHROPIC_API_KEY if not mock else "mock"
    result  = await get_reply(sender, message, api_key, mock_claude=mock)
    _record_lead(sender, message, result, True)
    return JSONResponse({"ok": True, **result})


@app.get("/test-ui", response_class=HTMLResponse)
async def test_ui():
    """Browser-based chat tester — open http://localhost:3000/test-ui"""
    return HTMLResponse(_TEST_UI_HTML)


_TEST_UI_HTML = """<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>בוט דלתות מיכאל — בדיקות</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#e5ddd5;height:100vh;display:flex;flex-direction:column}
#header{background:#075e54;color:#fff;padding:12px 16px;display:flex;align-items:center;gap:12px;flex-shrink:0}
#header h1{font-size:17px;font-weight:600;flex:1}
#sender-wrap{display:flex;align-items:center;gap:6px;font-size:13px}
#sender-input{padding:4px 8px;border-radius:4px;border:none;font-size:13px;width:180px;direction:ltr}
#btn-reset{background:#128c7e;color:#fff;border:none;padding:5px 10px;border-radius:4px;cursor:pointer;font-size:13px}
#btn-reset:hover{background:#0d6f63}
#mock-toggle{display:flex;align-items:center;gap:5px;font-size:12px;color:#d0ede9}
#mock-toggle input{cursor:pointer}
#chat{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:6px}
.msg-row{display:flex;max-width:75%}
.msg-row.user{align-self:flex-start;flex-direction:row-reverse}
.msg-row.bot{align-self:flex-end}
.bubble{padding:8px 12px;border-radius:8px;font-size:14px;line-height:1.5;white-space:pre-wrap;word-break:break-word;box-shadow:0 1px 2px rgba(0,0,0,.15)}
.user .bubble{background:#dcf8c6;border-bottom-left-radius:2px}
.bot .bubble{background:#fff;border-bottom-right-radius:2px}
.bubble.mock{background:#fff3cd;border-right:3px solid #ffc107}
.bubble.scripted{background:#e8f5e9;border-right:3px solid #4caf50}
.bubble.handoff{background:#e3f2fd;border-right:3px solid #2196f3}
.msg-time{font-size:10px;color:#999;margin-top:3px;text-align:left}
.msg-label{font-size:10px;padding:1px 5px;border-radius:10px;margin-bottom:2px;display:inline-block;align-self:flex-end}
.label-scripted{background:#c8e6c9;color:#1b5e20}
.label-mock{background:#fff3cd;color:#856404}
.label-claude{background:#e1f5fe;color:#01579b}
.label-handoff{background:#e3f2fd;color:#0d47a1;font-weight:600}
#meta{background:#f0f0f0;border-top:1px solid #ccc;padding:8px 16px;font-size:12px;color:#555;direction:ltr;min-height:28px;font-family:monospace}
#input-area{background:#f0f0f0;padding:8px 12px;display:flex;gap:8px;align-items:flex-end;flex-shrink:0}
#msg-input{flex:1;padding:10px 14px;border-radius:20px;border:none;font-size:14px;font-family:inherit;resize:none;max-height:120px;outline:none;direction:rtl}
#btn-send{background:#075e54;color:#fff;border:none;border-radius:50%;width:44px;height:44px;cursor:pointer;font-size:20px;display:flex;align-items:center;justify-content:center;flex-shrink:0}
#btn-send:hover{background:#128c7e}
#btn-send:disabled{background:#aaa;cursor:not-allowed}
.typing{color:#888;font-style:italic;font-size:13px;align-self:flex-end;padding:4px 12px}
.system-msg{text-align:center;font-size:12px;color:#888;background:rgba(255,255,255,.6);padding:4px 12px;border-radius:10px;align-self:center}
</style>
</head>
<body>
<div id="header">
  <h1>🚪 בוט דלתות מיכאל — ממשק בדיקות</h1>
  <label id="mock-toggle"><input type="checkbox" id="mock-cb" checked> מוק (ללא קרדיטים)</label>
  <div id="sender-wrap">
    <span>לקוח:</span>
    <input id="sender-input" value="test_1" placeholder="שם לקוח">
  </div>
  <button id="btn-reset">איפוס שיחה</button>
</div>
<div id="chat"></div>
<div id="meta">מוכן לבדיקה</div>
<div id="input-area">
  <textarea id="msg-input" rows="1" placeholder="כתוב הודעה..."></textarea>
  <button id="btn-send">➤</button>
</div>

<script>
const chat    = document.getElementById('chat');
const input   = document.getElementById('msg-input');
const sendBtn = document.getElementById('btn-send');
const meta    = document.getElementById('meta');
const mockCb  = document.getElementById('mock-cb');

function getSender(){
  const v = document.getElementById('sender-input').value.trim() || 'test_1';
  return v.replace(/[^a-zA-Z0-9_]/g,'_') + '@c.us';
}

function now(){
  return new Date().toLocaleTimeString('he-IL',{hour:'2-digit',minute:'2-digit'});
}

function addSystemMsg(txt){
  const d=document.createElement('div');
  d.className='system-msg';d.textContent=txt;
  chat.appendChild(d);scrollDown();
}

function addMsg(text, side, type=''){
  const row=document.createElement('div');
  row.className='msg-row '+side;
  let labelHtml='';
  if(type==='scripted') labelHtml='<div class="msg-label label-scripted">📋 תסריט</div>';
  else if(type==='mock')   labelHtml='<div class="msg-label label-mock">🤖 מוק</div>';
  else if(type==='claude') labelHtml='<div class="msg-label label-claude">✨ קלוד</div>';
  else if(type==='handoff')labelHtml='<div class="msg-label label-handoff">✅ הועבר לנציג</div>';
  const bubbleClass='bubble'+(type?' '+type:'');
  row.innerHTML=`
    <div>
      ${labelHtml}
      <div class="${bubbleClass}">${escHtml(text)}</div>
      <div class="msg-time">${now()}</div>
    </div>`;
  chat.appendChild(row);scrollDown();
}

function escHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scrollDown(){chat.scrollTop=chat.scrollHeight;}

function setMeta(obj){
  const parts=[];
  if(obj.summary) parts.push('📝 '+obj.summary);
  if(obj.service_type) parts.push('🚪 '+obj.service_type);
  if(obj.city) parts.push('📍 '+obj.city);
  if(obj.full_name) parts.push('👤 '+obj.full_name);
  if(obj.phone) parts.push('📞 '+obj.phone);
  if(obj.preferred_contact_hours) parts.push('⏰ '+obj.preferred_contact_hours);
  if(obj.handoff_to_human) parts.push('✅ HANDOFF');
  meta.textContent = parts.length ? parts.join(' | ') : 'אין מטאדאטה';
}

async function send(){
  const text=input.value.trim();
  if(!text) return;
  input.value='';input.style.height='';
  sendBtn.disabled=true;

  addMsg(text,'user');

  const typing=document.createElement('div');
  typing.className='typing';typing.textContent='נטלי מקלידה...';
  chat.appendChild(typing);scrollDown();

  try{
    const resp=await fetch('/test-chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sender:getSender(),message:text,mock:mockCb.checked})
    });
    const data=await resp.json();
    typing.remove();

    if(!data.ok){meta.textContent='שגיאה: '+(data.error||'?');return;}

    const isMock=data.reply_text&&data.reply_text.startsWith('🤖 [מוק');
    const isHandoff=data.handoff_to_human;
    const type = isHandoff?'handoff': isMock?'mock':'scripted_or_claude';

    // Determine if this is a scripted scenario response (doesn't start with 🤖)
    const finalType = isMock ? 'mock' : (isHandoff ? 'handoff' : (mockCb.checked ? 'scripted' : 'claude'));

    addMsg(data.reply_text, 'bot', finalType);
    if(data.reply_text_2){
      setTimeout(()=>{
        addMsg(data.reply_text_2,'bot', isMock?'mock':'scripted');
      },400);
    }
    setMeta(data);
    if(isHandoff) addSystemMsg('✅ הפרטים הועברו לנציג');
  }catch(e){
    typing.remove();
    meta.textContent='שגיאת רשת: '+e.message;
  }finally{
    sendBtn.disabled=false;
    input.focus();
  }
}

document.getElementById('btn-reset').onclick=async()=>{
  await fetch('/test-chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sender:getSender(),reset:true})});
  chat.innerHTML='';
  meta.textContent='שיחה אופסה';
  addSystemMsg('שיחה חדשה התחילה');
};

sendBtn.onclick=send;
input.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}
});
input.addEventListener('input',()=>{
  input.style.height='auto';
  input.style.height=Math.min(input.scrollHeight,120)+'px';
});

addSystemMsg('ממשק בדיקות — הודעות לא נשלחות לוואטסאפ');
</script>
</body>
</html>"""


def _check_admin(admin: str) -> Optional[JSONResponse]:
    """Return 403 response if ADMIN_SECRET is set and token doesn't match, else None."""
    if config.ADMIN_SECRET and admin != config.ADMIN_SECRET:
        return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)
    return None


@app.get("/diag", response_class=JSONResponse)
async def diag(admin: str = Query(default="")):
    """Lightweight diagnostics endpoint — runtime state, error counts, recent errors."""
    if (denied := _check_admin(admin)):
        return denied
    uptime_s = int(time.time() - _bot_start_time) if _bot_start_time else 0
    return {
        "uptime_seconds": uptime_s,
        "test_mode": config.TEST_MODE,
        "config": {
            "green_api_instance": config.GREEN_API_INSTANCE_ID[:4] + "...",
            "anthropic_key_set":  bool(config.ANTHROPIC_API_KEY),
            "sheets_configured":  bool(config.GOOGLE_SHEETS_WEBHOOK_URL),
            **_ROUTER_DIAG,
        },
        "tasks": {
            "poll_alive":    _poll_task is not None and not _poll_task.done(),
            "followup_alive": _followup_task is not None and not _followup_task.done(),
        },
        "conversations": {
            "active":      len(_conv_history),
            "total_turns": sum(len(v) for v in _conv_history.values()),
        },
        "followups": {
            "watching": sum(1 for s in _followup.values() if not s.get("closed")),
            "closed":   sum(1 for s in _followup.values() if s.get("closed")),
        },
        "dedup_cache_size": len(_processed_ids_set),
        "retry_queues": {
            "failed_sends": len(_failed_sends),
            "sheets_pending": len(_sheets_retry_queue),
        },
        "rate_limit": {
            "window_seconds": _RATE_WINDOW,
            "max_per_window": _RATE_MAX_MSG,
            "tracked_senders": len(_sender_msg_times),
        },
        "disk": {
            "data_dir": str(_DATA_DIR),
            "ephemeral_warning": not bool(config.DATA_DIR),
        },
        "error_counts": _error_counts,
        "recent_errors": list(_recent_errors),
    }


@app.get("/reload", response_class=JSONResponse)
async def reload_config(admin: str = Query(default="")):
    """Reload system prompt and FAQ from disk immediately — no restart needed."""
    if (denied := _check_admin(admin)):
        return denied
    t0 = time.time()
    await _refresh_system_prompt()
    await _refresh_faq()
    elapsed = round(time.time() - t0, 2)
    return {
        "ok": True,
        "reloaded_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elapsed_s": elapsed,
        "system_prompt_chars": _ROUTER_DIAG.get("system_prompt_chars"),
        "faq_count": _ROUTER_DIAG.get("faq_count"),
    }


@app.get("/test-sheets", response_class=JSONResponse)
async def test_sheets(admin: str = Query(default="")):
    """Send a test row to Google Sheets and report the result.
    Use this to verify the webhook URL is correct after updating it in Render."""
    if (denied := _check_admin(admin)):
        return denied
    if not config.GOOGLE_SHEETS_WEBHOOK_URL:
        return JSONResponse({"ok": False, "error": "GOOGLE_SHEETS_WEBHOOK_URL is not set in environment"}, status_code=400)

    url = config.GOOGLE_SHEETS_WEBHOOK_URL
    # Show partial URL for debugging (hide token if present)
    url_display = url[:60] + "..." if len(url) > 60 else url

    test_row = {
        "full_name":               "טסט בוט",
        "city":                    "נתיבות",
        "service_type":            "בדיקת חיבור — דלת כניסה מעוצבת",
        "datetime":                _utc_iso_to_il(datetime.utcnow().isoformat()),
        "preferred_contact_hours": "בוקר",
        "phone":                   "050-0000000",
        "notes":                   "שורת טסט — אפשר למחוק",
    }

    import httpx
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            r = await client.post(url, json=test_row)
        elapsed = round(time.time() - t0, 2)
        success = r.status_code in (200, 201, 302)
        return {
            "ok": success,
            "http_status": r.status_code,
            "elapsed_s": elapsed,
            "url_prefix": url_display,
            "row_sent": test_row,
            "response_body": r.text[:300] if r.text else None,
            "note": "302 = Apps Script success (normal). 200/201 also OK." if success else "Unexpected status — check URL and Apps Script deployment",
        }
    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        return JSONResponse({
            "ok": False,
            "error": str(exc)[:300],
            "elapsed_s": elapsed,
            "url_prefix": url_display,
        }, status_code=500)


@app.get("/test-ai", response_class=JSONResponse)
async def test_ai():
    """Fire a single real AI call and report which provider responded. No auth required."""
    from .engine import simple_router as _sr
    t0 = time.time()
    try:
        result = await _sr._call_ai(
            system="You are a test assistant. Reply with exactly: OK",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=100,
            api_key=config.ANTHROPIC_API_KEY,
            timeout=20.0,
        )
        elapsed = round(time.time() - t0, 2)
        return {
            "ok": True,
            "provider_used": _ROUTER_DIAG.get("last_ai_provider"),
            "response": result.strip()[:50],
            "elapsed_s": elapsed,
            "openrouter_key_set": _ROUTER_DIAG.get("openrouter_key_set"),
            "openrouter_failures": _ROUTER_DIAG.get("openrouter_failures", 0),
            "last_openrouter_error": _ROUTER_DIAG.get("last_openrouter_error"),
        }
    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        return JSONResponse({
            "ok": False,
            "error": str(exc)[:200],
            "elapsed_s": elapsed,
            "openrouter_key_set": _ROUTER_DIAG.get("openrouter_key_set"),
        }, status_code=500)


@app.get("/conversations", response_class=HTMLResponse)
async def conversations(test: str = "false", format: str = "html", admin: str = Query(default="")):
    if (denied := _check_admin(admin)):
        return denied
    is_test = test.lower() == "true"
    leads = _load_leads(is_test)
    entries = list(leads.values())

    if format == "json":
        return JSONResponse(leads)

    # ── Summary cards ─────────────────────────────────────────────────────────
    _STATUS_COLOR = {
        "הועבר לנציג":       "#e8f5e9",
        "סגירה בידידות":     "#e3f2fd",
        "נסגרה ללא מענה":    "#fff8e1",
    }
    _STATUS_BADGE = {
        "הועבר לנציג":       "#2e7d32",
        "סגירה בידידות":     "#1565c0",
        "נסגרה ללא מענה":    "#f57f17",
    }

    summary_cards = ""
    for lead in sorted(entries, key=lambda l: l.get("close_time", l.get("firstContact", "")), reverse=True):
        s = lead.get("conv_summary")
        if not s:
            continue
        phone_clean = lead["phone"].replace("@c.us", "")
        name = lead.get("full_name") or phone_clean
        reason = lead.get("close_reason", "—")
        close_dt = ""
        if lead.get("close_time"):
            try:
                close_dt = datetime.fromisoformat(lead["close_time"]).strftime("%d/%m/%Y %H:%M")
            except Exception:
                close_dt = lead["close_time"][:16]
        bg = _STATUS_COLOR.get(reason, "#fafafa")
        badge_color = _STATUS_BADGE.get(reason, "#666")
        summary_safe = s.replace("<", "&lt;").replace("\n", "<br>")
        summary_cards += f"""
  <div class="card" style="background:{bg}">
    <div class="card-header">
      <span class="name">{name}</span>
      <span class="phone">{phone_clean}</span>
      <span class="badge" style="background:{badge_color}">{reason}</span>
      <span class="dt">{close_dt}</span>
    </div>
    <div class="card-body">{summary_safe}</div>
  </div>"""

    # ── Message rows ──────────────────────────────────────────────────────────
    rows = ""
    for lead in entries:
        for m in lead.get("messages", []):
            is_bot = m["from"] == "bot"
            css = "bot" if is_bot else "customer"
            sender_label = "🤖 בוט" if is_bot else "👤 לקוח"
            text_safe = m["text"].replace("<", "&lt;")
            try:
                time_str = datetime.fromisoformat(m["time"]).strftime("%d/%m/%Y %H:%M")
            except Exception:
                time_str = m.get("time", "")[:16]
            rows += f'<tr class="{css}"><td>{lead["phone"].replace("@c.us","")}</td><td>{sender_label}</td><td style="white-space:pre-wrap">{text_safe}</td><td>{time_str}</td></tr>'

    total_msgs = sum(len(l.get("messages", [])) for l in entries)
    summarized = sum(1 for l in entries if l.get("conv_summary"))

    return f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
  <meta charset="UTF-8">
  <title>שיחות בוט - דלתות מיכאל</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 20px; background: #f0f2f5; }}
    h1 {{ color: #333; margin-bottom: 4px; }}
    h2 {{ color: #555; font-size: 16px; margin: 24px 0 10px; }}
    .meta {{ color: #888; font-size: 12px; margin-bottom: 16px; }}
    a {{ color: #075e54; }}
    .card {{ border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    .card-header {{ display: flex; gap: 12px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
    .name {{ font-weight: bold; font-size: 15px; }}
    .phone {{ color: #555; font-size: 13px; }}
    .badge {{ color: white; padding: 2px 9px; border-radius: 12px; font-size: 12px; }}
    .dt {{ color: #888; font-size: 12px; margin-right: auto; }}
    .card-body {{ font-size: 14px; line-height: 1.7; color: #333; white-space: pre-wrap; }}
    table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
             overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    th {{ background: #075e54; color: white; padding: 10px 14px; text-align: right; }}
    td {{ padding: 8px 14px; border-bottom: 1px solid #eee; vertical-align: top; max-width: 400px; }}
    tr.bot td {{ background: #dcf8c6; }}
    tr.customer td {{ background: #fff; }}
    tr:hover td {{ opacity: 0.85; }}
  </style>
</head>
<body>
  <h1>שיחות בוט — דלתות מיכאל</h1>
  <p class="meta">{len(entries)} לידים | {total_msgs} הודעות | {summarized} סיכומים
    &nbsp;|&nbsp; <a href="/conversations?test=false">פרודקשן</a>
    &nbsp;|&nbsp; <a href="/conversations?test=true">טסט</a>
    &nbsp;|&nbsp; <a href="/conversations?format=json">JSON</a>
  </p>

  <h2>📋 סיכומי פניות</h2>
  {summary_cards or '<p style="color:#999;font-size:14px">אין סיכומים עדיין — יופיעו כאשר פניות ייסגרו.</p>'}

  <h2>💬 כל ההודעות</h2>
  <table>
    <thead><tr><th>מספר</th><th>שולח</th><th>הודעה</th><th>זמן</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="4" style="text-align:center;color:#999">אין שיחות עדיין</td></tr>'}</tbody>
  </table>
</body>
</html>"""


@app.post("/inject-state", response_class=JSONResponse)
async def inject_state(request: Request, admin: str = Query(default="")):
    """Inject contact fields into an existing conversation state.
    Useful after a server restart wipes in-memory state mid-conversation.

    Body (JSON):
      sender   — phone number (with or without @c.us)
      full_name, phone, city, service_type, active_topics (list), preferred_contact_hours
      (all optional — only provided fields are updated)

    Usage: POST /inject-state?admin=<secret>
    """
    if (denied := _check_admin(admin)):
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON body"}, status_code=400)

    sender_raw = body.get("sender", "")
    if not sender_raw:
        return JSONResponse({"ok": False, "error": "sender required"}, status_code=400)

    chat_id = sender_raw if sender_raw.endswith("@c.us") else f"{sender_raw}@c.us"

    # Load or create state — use existing or start with empty dict
    state = _router_conv_state.get(chat_id) or {}

    # Apply provided fields
    updatable = ("full_name", "phone", "city", "service_type", "preferred_contact_hours",
                 "active_topics", "entrance_scope", "entrance_style", "entrance_model",
                 "interior_project_type", "interior_quantity", "interior_style",
                 "interior_model", "mamad_type", "customer_gender")
    updated = {}
    for field in updatable:
        if field in body and body[field] is not None:
            state[field] = body[field]
            updated[field] = body[field]

    _router_conv_state[chat_id] = state
    _save_conv_state()

    # Also persist in leads file
    leads = _load_leads(config.TEST_MODE)
    if chat_id not in leads:
        leads[chat_id] = {"phone": chat_id, "firstContact": datetime.utcnow().isoformat()}
    for field in ("full_name", "city", "service_type", "preferred_contact_hours"):
        if field in updated:
            leads[chat_id][field] = updated[field]
    if "phone" in updated:
        leads[chat_id]["callback_phone"] = updated["phone"]
    if "active_topics" in updated:
        leads[chat_id]["active_topics"] = updated["active_topics"]
    _save_leads(leads, config.TEST_MODE)

    logger.info("[ADMIN] State injected | sender=%s | fields=%s", chat_id, list(updated.keys()))
    return {"ok": True, "sender": chat_id, "updated_fields": updated, "state_snapshot": {
        k: state.get(k) for k in ("full_name", "phone", "city", "service_type",
                                   "active_topics", "preferred_contact_hours")
    }}


@app.get("/close-followup", response_class=JSONResponse)
async def close_followup(sender: str = Query(default=""), admin: str = Query(default="")):
    """Close the follow-up timer for a specific sender so no reminder is sent.
    Usage: /close-followup?sender=972535248428&admin=<secret>
    Sender can be with or without @c.us suffix.
    """
    if (denied := _check_admin(admin)):
        return denied
    if not sender:
        return JSONResponse({"ok": False, "error": "sender required"}, status_code=400)
    # Normalize — ensure @c.us suffix
    chat_id = sender if sender.endswith("@c.us") else f"{sender}@c.us"
    if chat_id in _followup:
        _followup[chat_id]["closed"] = True
        _save_followup()
        logger.info("[ADMIN] Follow-up closed manually | sender=%s", chat_id)
        return {"ok": True, "sender": chat_id, "action": "closed"}
    return {"ok": False, "sender": chat_id, "error": "sender not found in followup state"}


@app.get("/backfill-incomplete-leads", response_class=JSONResponse)
async def backfill_incomplete_leads(admin: str = Query(default="")):
    """
    One-shot backfill: scan conversations from the last 24 hours and send
    incomplete leads to Sheets for manual follow-up.

    Criteria for inclusion:
      • firstContact within last 24 hours
      • Last bot activity > 2 hours ago
      • At least one topic detected
      • Complete lead NOT already sent (sheets_sent=False)
      • Partial lead NOT already sent (partial_lead_sent=False)
    """
    if (denied := _check_admin(admin)):
        return denied
    if not config.GOOGLE_SHEETS_WEBHOOK_URL:
        return JSONResponse({"ok": False, "error": "GOOGLE_SHEETS_WEBHOOK_URL not set"}, status_code=400)

    now         = time.time()
    cutoff_24h  = now - 24 * 3600
    cutoff_2h   = now -  2 * 3600

    leads      = _load_leads(config.TEST_MODE)
    processed  = []
    skipped    = []

    # All candidate senders: from leads file + active conv history in memory
    candidates = set(leads.keys()) | set(_conv_history.keys())

    for sender in candidates:
        lead_rec   = leads.get(sender, {})
        conv_state = _router_conv_state.get(sender, {})

        # Filter: only last 24 hours (based on firstContact)
        first_contact_str = lead_rec.get("firstContact", "")
        try:
            first_contact_ts = datetime.fromisoformat(first_contact_str).timestamp()
        except Exception:
            first_contact_ts = 0.0
        if first_contact_ts < cutoff_24h:
            skipped.append({"sender": sender[-10:], "reason": "older than 24h"})
            continue

        # Filter: already in Sheets
        if lead_rec.get("sheets_sent") or lead_rec.get("partial_lead_sent"):
            skipped.append({"sender": sender[-10:], "reason": "already in sheets"})
            continue

        # Filter: at least one topic detected
        active_topics = conv_state.get("active_topics") or []
        if not active_topics:
            skipped.append({"sender": sender[-10:], "reason": "no topics detected"})
            continue

        # Filter: last bot activity > 2 hours ago
        followup_state = _followup.get(sender, {})
        last_activity  = followup_state.get("last_bot_time") or first_contact_ts
        if last_activity > cutoff_2h:
            skipped.append({"sender": sender[-10:], "reason": "activity < 2h ago"})
            continue

        # Eligible — send partial lead to Sheets
        try:
            await _send_incomplete_lead_to_sheets(sender, config.TEST_MODE)
            processed.append({
                "sender": sender[-10:],
                "topics": active_topics,
                "name":   conv_state.get("full_name") or "",
            })
            logger.info("[BACKFILL] Sent | sender=%s | topics=%s", sender, active_topics)
        except Exception as exc:
            skipped.append({"sender": sender[-10:], "reason": f"error: {exc}"})
            logger.error("[BACKFILL_ERR] sender=%s | %s", sender, exc)

    return JSONResponse({
        "ok":        True,
        "sent":      len(processed),
        "skipped":   len(skipped),
        "processed": processed,
        "skip_log":  skipped,
    })
