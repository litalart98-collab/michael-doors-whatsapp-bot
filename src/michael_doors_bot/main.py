"""
FastAPI server — webhook mode (Green API sends POST on each incoming message).
Polling loop runs as fallback if webhook is not configured.
"""
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config
from .engine.simple_router import (
    _conversations as _conv_history,
    clear_conversation,
    generate_conversation_summary,
    get_closing_message,
    get_followup_message,
    get_reply,
    is_closing_intent,
)
from .providers.greenapi import GreenAPIClient
from .providers.google_sheets import append_lead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT            = Path(__file__).parent.parent.parent
_LEADS_FILE      = _ROOT / "leads.json"
_TEST_LEADS_FILE = _ROOT / "leads_test.json"
_SESSIONS_FILE   = _ROOT / "sessions.json"

SESSION_TIMEOUT       = 30 * 60  # seconds
FOLLOWUP_DELAY        = 15 * 60  # 15 min silence → send follow-up
CLOSE_AFTER_FOLLOWUP  =  7 * 60  # 7 min after follow-up → close inquiry

green = GreenAPIClient(config.GREEN_API_INSTANCE_ID, config.GREEN_API_TOKEN, config.GREEN_API_URL)

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
    """Generate a summary and attach it to the lead record."""
    try:
        summary = await generate_conversation_summary(sender, config.ANTHROPIC_API_KEY)
        leads = _load_leads(is_test)
        if sender in leads:
            leads[sender]["conv_summary"] = summary
            leads[sender]["close_reason"] = close_reason
            leads[sender]["close_time"] = datetime.utcnow().isoformat()
            _save_leads(leads, is_test)
            logger.info("Summary saved | sender=%s | reason=%s", sender, close_reason)
    except Exception as exc:
        logger.error("_attach_summary error | sender=%s | %s", sender, exc)


def _record_lead(sender: str, user_msg: str, result: dict, is_test: bool) -> None:
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


async def _maybe_send_to_sheets(lead: dict, result: dict) -> None:
    if not config.GOOGLE_SHEETS_WEBHOOK_URL:
        return
    if not result.get("handoff_to_human"):
        return
    if lead.get("sheets_sent"):
        return
    row = {
        "full_name":               lead.get("full_name", ""),
        "city":                    lead.get("city", ""),
        "service_type":            lead.get("service_type", ""),
        "datetime":                lead.get("firstContact", ""),
        "preferred_contact_hours": lead.get("preferred_contact_hours", ""),
        "phone":                   lead.get("phone", ""),
    }
    await append_lead(config.GOOGLE_SHEETS_WEBHOOK_URL, row)
    lead["sheets_sent"] = True


# ── Follow-up tracker ─────────────────────────────────────────────────────────
# {sender: {"last_bot_time": float, "followup_sent": bool, "followup_time": float, "closed": bool}}
_followup: dict[str, dict] = {}

_CLOSE_MSG = (
    "מכיוון שלא קיבלנו תגובה, נסגור את הפנייה לעת עתה 🙂\n"
    "אם תרצו לחזור אלינו בכל עת — אנחנו כאן!\n"
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


def _followup_mark_bot_replied(sender: str) -> None:
    if sender not in _followup:
        _followup[sender] = {"last_bot_time": time.time(), "followup_sent": False, "followup_time": 0.0, "closed": False}
    else:
        _followup[sender]["last_bot_time"] = time.time()
        _followup[sender]["closed"] = False


async def _followup_loop() -> None:
    logger.info("Follow-up loop started")
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for sender, state in list(_followup.items()):
            if not _is_individual_chat(sender):
                continue
            if state.get("closed"):
                continue
            last_bot = state.get("last_bot_time", 0.0)
            followup_sent = state.get("followup_sent", False)
            followup_time = state.get("followup_time", 0.0)

            if not followup_sent and now - last_bot >= FOLLOWUP_DELAY:
                try:
                    msg = await get_followup_message(sender, config.ANTHROPIC_API_KEY)
                    await green.send_message(sender, msg)
                    state["followup_sent"] = True
                    state["followup_time"] = time.time()
                    logger.info("Follow-up sent | sender=%s", sender)
                except Exception as exc:
                    logger.error("Follow-up error | sender=%s | %s", sender, exc)

            elif followup_sent and now - followup_time >= CLOSE_AFTER_FOLLOWUP:
                try:
                    await green.send_message(sender, _CLOSE_MSG)
                    state["closed"] = True
                    logger.info("Inquiry closed | sender=%s", sender)
                    await _attach_summary(sender, "נסגרה ללא מענה", False)
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


async def _process_message(sender: str, text: str) -> None:
    try:
        if not _is_individual_chat(sender):
            logger.info("Skipping non-individual chat | sender=%s", sender)
            return
        if config.TEST_MODE:
            if sender != config.TEST_PHONE:
                logger.info("TEST_MODE: blocked sender=%s (allowed=%s)", sender, config.TEST_PHONE)
                return
            if text.strip() == "#reset":
                clear_conversation(sender)
                _followup.pop(sender, None)
                await green.send_message(sender, "שיחה אופסה ✓")
                return

        # Customer replied — reset follow-up timer
        _followup_reset(sender)

        # Smart closing: if customer says goodbye/thanks mid-conversation, close gracefully
        conv_turns = len(_conv_history.get(sender, []))
        if is_closing_intent(text, conv_turns):
            logger.info("Closing intent detected | sender=%s", sender)
            closing_msg = await get_closing_message(sender, config.ANTHROPIC_API_KEY)
            try:
                await green.send_message(sender, closing_msg)
                _followup[sender] = {
                    "last_bot_time": time.time(),
                    "followup_sent": True,
                    "followup_time": time.time(),
                    "closed": True,
                }
                logger.info("Conversation closed gracefully | sender=%s", sender)
                await _attach_summary(sender, "סגירה בידידות", config.TEST_MODE)
            except Exception as send_err:
                logger.error("Closing send failed | sender=%s | %s", sender, send_err)
            return

        result = await get_reply(sender, text, config.ANTHROPIC_API_KEY)
        logger.info("Got reply for %s: %s", sender, result["reply_text"][:60])
        lead = _record_lead(sender, text, result, config.TEST_MODE)
        await _maybe_send_to_sheets(lead, result)
        if result.get("handoff_to_human"):
            await _attach_summary(sender, "הועבר לנציג", config.TEST_MODE)
        try:
            await green.send_message(sender, result["reply_text"])
            _followup_mark_bot_replied(sender)
            logger.info("Message sent to %s", sender)
        except Exception as send_err:
            logger.error("Send failed | sender=%s | %s", sender, send_err)
    except Exception as exc:
        logger.error("_process_message error | sender=%s | %s", sender, exc)


# ── Polling loop (fallback when no webhook configured) ────────────────────────
async def _poll_loop() -> None:
    logger.info("Polling loop started (fallback mode)")
    while True:
        try:
            notification = await green.receive_notification()
            if not notification:
                await asyncio.sleep(2)
                continue

            receipt_id = notification.get("receiptId")
            body = notification.get("body", {})

            if body.get("typeWebhook") == "incomingMessageReceived":
                sender = body.get("senderData", {}).get("chatId", "")
                msg_data = body.get("messageData", {})
                text = (
                    msg_data.get("textMessageData", {}).get("textMessage")
                    or msg_data.get("extendedTextMessageData", {}).get("text")
                    or ""
                )
                if sender and text:
                    logger.info("Poll incoming | sender=%s | text=%s", sender, text[:60])
                    await _process_message(sender, text)

            if receipt_id:
                await green.delete_notification(receipt_id)

        except Exception as exc:
            logger.error("Poll loop error: %s", exc)
            await asyncio.sleep(5)


# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("=== BOT STARTING ===")
    logger.info("TEST_MODE=%s | TEST_PHONE=%s", config.TEST_MODE, config.TEST_PHONE or "(not set)")
    logger.info("GREEN_API_INSTANCE=%s", config.GREEN_API_INSTANCE_ID)
    poll_task    = asyncio.create_task(_poll_loop())
    followup_task = asyncio.create_task(_followup_loop())
    yield
    poll_task.cancel()
    followup_task.cancel()


app = FastAPI(lifespan=_lifespan)


@app.get("/", response_class=JSONResponse)
async def health():
    return {"status": "Bot is running", "mode": "webhook+polling"}


@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)

    logger.info("Webhook received: typeWebhook=%s", body.get("typeWebhook"))

    if body.get("typeWebhook") != "incomingMessageReceived":
        return JSONResponse({"ok": True})

    sender = body.get("senderData", {}).get("chatId", "")
    msg_data = body.get("messageData", {})
    text = (
        msg_data.get("textMessageData", {}).get("textMessage")
        or msg_data.get("extendedTextMessageData", {}).get("text")
        or ""
    )

    if sender and text:
        logger.info("Webhook incoming | sender=%s | text=%s", sender, text[:60])
        try:
            await asyncio.wait_for(_process_message(sender, text), timeout=60.0)
            leads = _load_leads(config.TEST_MODE)
            logger.info("After processing — leads count: %d", len(leads))
        except asyncio.TimeoutError:
            logger.error("Webhook timeout for sender=%s", sender)
        except Exception as exc:
            logger.error("Webhook processing error: %s", exc)

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


@app.get("/conversations", response_class=HTMLResponse)
async def conversations(test: str = "false", format: str = "html"):
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
                pass
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
            time_str = datetime.fromisoformat(m["time"]).strftime("%d/%m/%Y %H:%M")
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
