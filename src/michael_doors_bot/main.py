"""
FastAPI server — webhook mode (Green API sends POST on each incoming message).
Polling loop runs as fallback if webhook is not configured.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import config
from .engine.simple_router import clear_conversation, get_reply
from .providers.greenapi import GreenAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT            = Path(__file__).parent.parent.parent
_LEADS_FILE      = _ROOT / "leads.json"
_TEST_LEADS_FILE = _ROOT / "leads_test.json"
_SESSIONS_FILE   = _ROOT / "sessions.json"

SESSION_TIMEOUT  = 30 * 60  # seconds

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
    if result.get("handoff_to_human"):
        lead["handoff_to_human"] = True
        lead["handoff_time"] = datetime.utcnow().isoformat()
    if result.get("summary"):
        lead["summary"] = result["summary"]

    lead["messages"].append({"from": "customer", "text": user_msg, "time": datetime.utcnow().isoformat()})
    if result.get("reply_text"):
        lead["messages"].append({"from": "bot", "text": result["reply_text"], "time": datetime.utcnow().isoformat()})

    _save_leads(leads, is_test)


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
    import time
    ts = _sessions.get(sender)
    if ts is None:
        return False
    if time.time() - ts > SESSION_TIMEOUT:
        del _sessions[sender]
        _save_sessions()
        return False
    return True


def _open_session(sender: str) -> None:
    import time
    _sessions[sender] = time.time()
    _save_sessions()
    clear_conversation(sender)
    logger.info("TEST SESSION | Opened | %s", sender)


def _touch_session(sender: str) -> None:
    import time
    _sessions[sender] = time.time()
    _save_sessions()


def _close_session(sender: str) -> None:
    _sessions.pop(sender, None)
    _save_sessions()
    logger.info("TEST SESSION | Closed by #endtest | %s", sender)


# ── Message processor ─────────────────────────────────────────────────────────
async def _process_message(sender: str, text: str) -> None:
    try:
        if config.TEST_MODE:
            if sender != config.TEST_PHONE:
                return
            if text.strip() == "#test":
                _open_session(sender)
                await green.send_message(sender, "מצב טסט הופעל. שלח הודעה כלשהי להתחיל.")
                return
            if text.strip() == "#endtest":
                _close_session(sender)
                await green.send_message(sender, "מצב טסט הסתיים.")
                return
            if not _has_active_session(sender):
                return
            _touch_session(sender)

        result = await get_reply(sender, text, config.ANTHROPIC_API_KEY)
        logger.info("Sending reply to %s: %s", sender, result["reply_text"][:60])
        await green.send_message(sender, result["reply_text"])
        _record_lead(sender, text, result, config.TEST_MODE)
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
    task = asyncio.create_task(_poll_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=_lifespan)


@app.get("/", response_class=JSONResponse)
async def health():
    return {"status": "Bot is running", "mode": "webhook+polling"}


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
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
        background_tasks.add_task(_process_message, sender, text)

    return JSONResponse({"ok": True})


@app.get("/conversations", response_class=HTMLResponse)
async def conversations(test: str = "false", format: str = "html"):
    is_test = test.lower() == "true"
    leads = _load_leads(is_test)
    entries = list(leads.values())

    if format == "json":
        return JSONResponse(leads)

    rows = ""
    for lead in entries:
        for m in lead.get("messages", []):
            is_bot = m["from"] == "bot"
            css = "bot" if is_bot else "customer"
            sender_label = "🤖 בוט" if is_bot else "👤 לקוח"
            text_safe = m["text"].replace("<", "&lt;")
            time_str = datetime.fromisoformat(m["time"]).strftime("%d/%m/%Y %H:%M")
            rows += f'<tr class="{css}"><td>{lead["phone"]}</td><td>{sender_label}</td><td style="white-space:pre-wrap">{text_safe}</td><td>{time_str}</td></tr>'

    total_msgs = sum(len(l.get("messages", [])) for l in entries)

    return f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
  <meta charset="UTF-8">
  <title>שיחות בוט - דלתות מיכאל</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
    h1 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    th {{ background: #075e54; color: white; padding: 10px 14px; text-align: right; }}
    td {{ padding: 8px 14px; border-bottom: 1px solid #eee; vertical-align: top; max-width: 400px; }}
    tr.bot td {{ background: #dcf8c6; }}
    tr.customer td {{ background: #fff; }}
    tr:hover td {{ opacity: 0.85; }}
    .meta {{ color: #888; font-size: 12px; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>שיחות בוט — דלתות מיכאל</h1>
  <p class="meta">{len(entries)} לידים | {total_msgs} הודעות
    &nbsp;|&nbsp; <a href="/conversations?test=false">פרודקשן</a>
    &nbsp;|&nbsp; <a href="/conversations?test=true">טסט</a>
    &nbsp;|&nbsp; <a href="/conversations?format=json">JSON</a>
  </p>
  <table>
    <thead><tr><th>מספר</th><th>שולח</th><th>הודעה</th><th>זמן</th></tr></thead>
    <tbody>{rows or '<tr><td colspan="4" style="text-align:center;color:#999">אין שיחות עדיין</td></tr>'}</tbody>
  </table>
</body>
</html>"""
