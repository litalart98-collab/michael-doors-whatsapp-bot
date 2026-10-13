"""
key_message.py — Minimal, deterministic "key message" flow (pure, no AI, no I/O).

The simplest safe auto-responder for inbound WhatsApp leads (Green API):

  Turn 1  (first inbound message)  → send the KEY MESSAGE (qualifying opener).
  Turn 2  (customer's reply)       → send the HANDOFF message, store the reply,
                                      hand off to a human and go silent.
  Later   (any further message)    → silent (the human rep continues manually).

`process(state, text)` is a PURE function — no network, no disk, no clock. The
caller performs the side effects it signals via `intents`.
"""
from __future__ import annotations

from typing import Optional

from . import flow_config as C
from .flow_machine import is_agent_request

# ── Steps ─────────────────────────────────────────────────────────────────────
START          = "START"
AWAITING_REPLY = "AWAITING_REPLY"
DONE           = "DONE"

# ── Fixed messages (approved wording) ─────────────────────────────────────────
KEY_MESSAGE = (
    "שלום 👋 תודה שפניתם לדלתות מיכאל — זהו מענה אוטומטי.\n"
    "כדי שנחזור אליכם בהקדם עם כל הפרטים, נשמח לדעת:\n"
    "איזה סוג דלת מעניין אתכם (דלת כניסה / דלת פנים / ממ\"ד / אחר), כמה דלתות, ומאיפה אתם?"
)

HANDOFF_MESSAGE = (
    "תודה! הפנייה מועברת לנציג אנושי, ניצור איתכם קשר בימים הקרובים!"
)


def new_state() -> dict:
    return {
        "currentStep":      START,
        "conversationMode": C.MODE_BOT_COLLECTING,
        "botEnabled":       True,
        "collectedData":    {},
        "handoffReason":    None,
    }


def _result(state, replies, *, airtable=None, notify_owner=False, terminal=False) -> dict:
    if isinstance(replies, str):
        replies = [replies]
    return {
        "state": state,
        "replies": list(replies),
        "intents": {
            "airtable": airtable,          # None | "sync" | "complete"
            "notify_owner": notify_owner,
            "terminal": terminal,
        },
    }


def _finish(state: dict, reason: str = C.HANDOFF_READY) -> dict:
    """Send the handoff message, go silent, hand off to a human."""
    state["conversationMode"] = C.MODE_WAITING_FOR_AGENT
    state["currentStep"] = DONE
    state["botEnabled"] = False
    state["handoffReason"] = reason
    return _result(state, HANDOFF_MESSAGE, airtable="complete", notify_owner=True, terminal=True)


def process(state: Optional[dict], text: str, is_media: bool = False) -> dict:
    """Advance the key-message flow by one inbound message. Pure function."""
    state = dict(state) if state else new_state()
    state["collectedData"] = dict(state.get("collectedData") or {})
    step = state.get("currentStep", START)

    # Already handed off / human active → stay silent.
    if not state.get("botEnabled", True) or state.get("conversationMode") in (
        C.MODE_WAITING_FOR_AGENT, C.MODE_HUMAN_ACTIVE
    ):
        return _result(state, [])

    text = text or ""

    # Explicit agent request at any active step → hand off immediately.
    if not is_media and is_agent_request(text):
        return _finish(state, C.HANDOFF_CUSTOMER_REQUESTED)

    if step == START:
        # First inbound message → send the key message, wait for the reply.
        state["currentStep"] = AWAITING_REPLY
        return _result(state, KEY_MESSAGE, airtable="sync")

    if step == AWAITING_REPLY:
        # Store the customer's qualifying answer (where from / how many doors).
        if not is_media and text.strip():
            state["collectedData"]["lead_reply"] = text.strip()
        return _finish(state)

    # DONE or unknown → silent.
    return _result(state, [])
