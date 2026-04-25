"""
FastAPI server — webhook mode (Green API sends POST on each incoming message).
Polling loop runs as fallback if webhook is not configured.
"""
import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config
from .engine.simple_router import (
    DIAG_STATE as _ROUTER_DIAG,
    ERROR_REPLIES,
    _conversations as _conv_history,
    clear_conversation,
    generate_conversation_summary,
    get_closing_message,
    get_followup_message,
    get_reply,
    is_closing_intent,
    is_working_hours,
    _refresh_system_prompt,
    _refresh_faq,
)
from .providers.greenapi import GreenAPIClient
from .providers.google_sheets import append_lead
from .providers.supabase_store import upsert_lead, save_followup, load_all_conversations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT     = Path(__file__).parent.parent.parent
# DATA_DIR can be set to a Render Persistent Disk mount (e.g. /data) so runtime
# data files survive service restarts.  Falls back to project root.
_DATA_DIR        = Path(config.DATA_DIR) if config.DATA_DIR else _ROOT
_LEADS_FILE      = _DATA_DIR / "leads.json"
_TEST_LEADS_FILE = _DATA_DIR / "leads_test.json"
_SESSIONS_FILE   = _DATA_DIR / "sessions.json"
_DEDUP_FILE      = _DATA_DIR / "dedup_ids.json"   # persisted dedup cache
_FOLLOWUP_FILE   = _DATA_DIR / "followup_state.json"  # persisted follow-up timers

SESSION_TIMEOUT       = 30 * 60  # seconds
FOLLOWUP_DELAY        = 15 * 60  # 15 min silence → send follow-up
CLOSE_AFTER_FOLLOWUP  =  7 * 60  # 7 min after follow-up → close inquiry

_BOT_ERROR_MSG    = "רגע, בודקת 😊 תכתוב לי שוב בעוד רגע ואענה לך"
_CONTACT_FALLBACK = "תודה, קיבלנו את ההודעה שלכם. ניצור איתכם קשר בהקדם להמשך טיפול."
_NON_TEXT_MSG     = "שלום 😊 אני יכולה לעזור רק עם הודעות טקסט. במה אפשר לעזור?"

# Max text length passed into _process_message — truncated if exceeded
_MAX_MSG_CHARS = 2000

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
    if result.get("handoff_to_human"):
        lead["handoff_to_human"] = True
        lead["handoff_time"] = datetime.utcnow().isoformat()
    if result.get("summary"):
        lead["summary"] = result["summary"]

    lead["messages"].append({"from": "customer", "text": user_msg, "time": datetime.utcnow().isoformat()})
    if result.get("reply_text"):
        lead["messages"].append({"from": "bot", "text": result["reply_text"], "time": datetime.utcnow().isoformat()})

    _save_leads(leads, is_test)
    return lead


async def _maybe_send_to_sheets(lead: dict, result: dict, is_test: bool) -> None:
    if not config.GOOGLE_SHEETS_WEBHOOK_URL:
        return
    if not result.get("handoff_to_human"):
        return
    if lead.get("sheets_sent"):
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

    svc = lead.get("service_type", "")
    cnt = lead.get("doors_count")
    service_field = f"{svc} — {cnt} יחידות" if svc and cnt else svc
    row = {
        "full_name":               lead.get("full_name", ""),
        "city":                    lead.get("city", ""),
        "service_type":            service_field,
        "datetime":                lead.get("firstContact", ""),
        "preferred_contact_hours": lead.get("preferred_contact_hours", ""),
        "phone":                   phone_clean,
    }
    sender_id = lead.get("phone", "")
    try:
        await append_lead(config.GOOGLE_SHEETS_WEBHOOK_URL, row)
        # Only mark as sent after confirmed success
        lead["sheets_sent"] = True
        if sender_id:
            fresh = _load_leads(is_test)
            if sender_id in fresh:
                fresh[sender_id]["sheets_sent"] = True
                _save_leads(fresh, is_test)
    except Exception as exc:
        _record_error("send_fail", sender_id, f"sheets initial: {exc}")
        _queue_sheets_retry(row, is_test, sender_id, exc)


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
    "ראיתי שלא חזרת, אז נסגור את הפנייה לעת עתה 😊\n"
    "אם תרצה/י לחזור בכל שעה ולהתחיל שיחה חדשה — נשמח לעזור!\n"
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
                # Never send follow-ups outside business hours — defer until opening time
                if not is_working_hours():
                    continue

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
                # Never send outside business hours — defer until opening time
                if not is_working_hours():
                    continue

                try:
                    await green.send_message(sender, _CLOSE_MSG)
                    state["closed"] = True
                    _save_followup()
                    logger.info("[BOT:CLOSE] Inquiry closed (no-response) | sender=%s", sender)
                    await _attach_summary(sender, "נסגרה ללא מענה", config.TEST_MODE)
                except Exception as exc:
                    logger.error("Close message error | sender=%s | %s", sender, exc)


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
                logger.info("[BOT:CLOSE] Closing intent | sender=%s", sender)
                closing_msg = await get_closing_message(sender, config.ANTHROPIC_API_KEY)
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
                if sender and msg_id and _is_duplicate(msg_id):
                    logger.info("[BOT:DEDUP] Poll duplicate skipped | id=%s | sender=%s", msg_id, sender)
                elif sender and not text and _is_individual_chat(sender):
                    msg_type = msg_data.get("typeMessage", "")
                    if msg_type and msg_type not in ("textMessage", "extendedTextMessage", ""):
                        logger.info("[BOT:NON_TEXT] type=%s | sender=%s", msg_type, sender)
                        _followup_reset(sender)
                        try:
                            await green.send_message(sender, _NON_TEXT_MSG)
                            _followup_mark_bot_replied(sender)
                        except Exception as exc:
                            _record_error("send_fail", sender, str(exc))
                            logger.error("[BOT:SEND_FAIL] Non-text reply | sender=%s | %s", sender, exc)
                elif sender and text:
                    _track_msg_id(msg_id)
                    logger.info("[BOT:RECV] Poll | sender=%s | text=%s", sender, text[:60])
                    await _process_message(sender, text)

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
_poll_task: asyncio.Task | None = None
_followup_task: asyncio.Task | None = None
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
            _followup_reset(sender)  # customer is active — reset the follow-up clock
            try:
                await green.send_message(sender, _NON_TEXT_MSG)
                _followup_mark_bot_replied(sender)
            except Exception as exc:
                _record_error("send_fail", sender, str(exc))
                logger.error("[BOT:SEND_FAIL] Non-text reply | sender=%s | %s", sender, exc)
        return JSONResponse({"ok": True})

    if sender and text:
        logger.info("[BOT:RECV] Webhook | sender=%s | chars=%d | text=%s", sender, len(text), text[:60])
        try:
            await asyncio.wait_for(_process_message(sender, text), timeout=60.0)
        except asyncio.TimeoutError:
            _record_error("webhook", sender, "60s processing timeout")
            logger.error("[BOT:TIMEOUT] Webhook timeout | sender=%s — sending fallback", sender)
            try:
                await green.send_message(sender, _BOT_ERROR_MSG)
            except Exception:
                pass
        except Exception as exc:
            _record_error("webhook", sender, str(exc))
            logger.error("[BOT:WEBHOOK_ERR] sender=%s | %s", sender, exc)
            try:
                await green.send_message(sender, _BOT_ERROR_MSG)
            except Exception:
                pass

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


def _check_admin(admin: str) -> JSONResponse | None:
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


@app.get("/test-ai", response_class=JSONResponse)
async def test_ai(admin: str = Query(default="")):
    """Fire a single real AI call and report which provider responded."""
    if (denied := _check_admin(admin)):
        return denied
    from .engine import simple_router as _sr
    t0 = time.time()
    try:
        result = await _sr._call_ai(
            system="You are a test assistant. Reply with exactly: OK",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10,
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
