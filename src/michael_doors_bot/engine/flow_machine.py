"""
flow_machine.py — The DETERMINISTIC bot engine (pure, no AI, no I/O).

`process(state, text)` is a PURE function: given the current conversation state
and the customer's message, it returns the next state plus the messages to send.
It never calls an AI model, never guesses, never does fuzzy matching, and never
performs I/O (no network, no disk, no clock). Persistence and side effects
(Airtable sync, owner notification, timestamps) are the caller's job — the
engine only *signals* them via the returned `intents`.

This makes the whole conversation logic exhaustively unit-testable and
100% reproducible.
"""
from __future__ import annotations

import re
from typing import Optional

from . import flow_config as C


# ══════════════════════════════════════════════════════════════════════════════
# Normalization (deterministic — no fuzzy matching)
# ══════════════════════════════════════════════════════════════════════════════
def normalize(text: str) -> str:
    """Lowercase, trim, collapse whitespace, strip trivial punctuation, and drop
    Hebrew gershayim/quotes so that ממ"ד and ממד compare equal."""
    if not text:
        return ""
    t = text.strip()
    t = t.replace('"', "").replace("'", "").replace("״", "").replace("׳", "")
    t = re.sub(r"[.,!?;:–—]+", " ", t)   # trivial punctuation → space
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _norm_table(raw: dict) -> dict:
    """Normalize the keys of an option table once, for exact-match lookup."""
    return {normalize(k): v for k, v in raw.items()}


# Pre-normalized lookups (built once at import).
_MENU = {step: (msg, _norm_table(opts)) for step, (msg, opts) in C.MENU_STEPS.items()}
_CONFIRM = _norm_table(C.OPTIONS_CONFIRM)
_AGENT_ALIASES = [normalize(a) for a in C.AGENT_ALIASES]
_PRICE_ALIASES = [normalize(a) for a in C.PRICE_ALIASES]


def _match_menu(step: str, text: str) -> Optional[str]:
    """Return the canonical value for a menu step, or None if not an exact match."""
    _, table = _MENU[step]
    return table.get(normalize(text))


def _match_confirm(text: str) -> Optional[str]:
    return _CONFIRM.get(normalize(text))


def is_agent_request(text: str) -> bool:
    n = normalize(text)
    return any(alias and alias in n for alias in _AGENT_ALIASES)


def is_price_question(text: str) -> bool:
    n = normalize(text)
    return any(alias and alias in n for alias in _PRICE_ALIASES)


# ══════════════════════════════════════════════════════════════════════════════
# Field validation / normalization (contact details)
# ══════════════════════════════════════════════════════════════════════════════
_NAME_RE = re.compile(r"[A-Za-z֐-׿\-\s]{2,}$")
_LETTER_RE = re.compile(r"[A-Za-z֐-׿]")


def validate_name(value: Optional[str]) -> Optional[str]:
    """≥2 letters, Hebrew/English/space/hyphen only. Never invents a surname."""
    if not value:
        return None
    v = value.strip()
    if not _NAME_RE.match(v):
        return None
    if len(_LETTER_RE.findall(v)) < 2:
        return None
    return v


def validate_city(value: Optional[str]) -> Optional[str]:
    """Short text; reject a digits-only answer."""
    if not value:
        return None
    v = value.strip()
    if not v:
        return None
    if re.fullmatch(r"\d+", v.replace(" ", "")):
        return None
    return v


def normalize_phone(value: Optional[str]) -> Optional[str]:
    """Normalize an Israeli mobile to canonical 0XXXXXXXXX. No guessing/correcting.

    Accepts: 05XXXXXXXX, 05X-XXXXXXX, 9725XXXXXXXX, +9725XXXXXXXX.
    Returns None if it does not match a valid pattern.
    """
    if not value:
        return None
    p = value.strip().replace("-", "").replace(" ", "")
    if p.startswith("+972"):
        p = "0" + p[4:]
    elif p.startswith("972"):
        p = "0" + p[3:]
    if re.fullmatch(r"0\d{9}", p) and p.startswith("05"):
        return p
    return None


def parse_contact(text: str) -> dict:
    """Extract fields ONLY by their labels (no AI). Returns raw (unvalidated) values."""
    out: dict = {}
    labels = (
        ("שם מלא:", "fullName"),
        ("עיר:", "city"),
        ("מספר טלפון:", "contactPhone"),
        ("טלפון:", "contactPhone"),
    )
    for line in text.splitlines():
        s = line.strip()
        for label, key in labels:
            if s.startswith(label) and key not in out:
                out[key] = s[len(label):].strip()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════
def new_state() -> dict:
    return {
        "currentStep":      C.START,
        "conversationMode": C.MODE_BOT_COLLECTING,
        "botEnabled":       True,
        "invalidAttempts":  0,
        "collectedData":    {},
        "savedQuestions":   [],
        "handoffReason":    None,
    }


def _result(state: dict, replies, *, airtable=None, notify_owner=False, terminal=False) -> dict:
    if isinstance(replies, str):
        replies = [replies]
    return {
        "state": state,
        "replies": list(replies),
        "intents": {
            "airtable": airtable,          # None | "sync" | "complete"
            "notify_owner": notify_owner,  # send owner notification once
            "terminal": terminal,          # bot is now silent
        },
    }


def build_summary(collected: dict) -> str:
    inquiry = collected.get("inquiryType")
    lines = ["תודה, אלה הפרטים שקיבלנו:", ""]
    lines.append("נושא הפנייה: " + C.LABELS_INQUIRY_TYPE.get(inquiry, ""))
    if inquiry == "ENTRANCE_DOOR" and collected.get("frameType"):
        lines.append("משקוף: " + C.LABELS_FRAME.get(collected["frameType"], ""))
    if inquiry == "INTERIOR_DOORS" and collected.get("quantityRange"):
        lines.append("כמות דלתות: " + C.LABELS_QUANTITY.get(collected["quantityRange"], ""))
    if inquiry == "OTHER" and collected.get("otherInquiry"):
        lines.append("פירוט: " + collected["otherInquiry"])
    lines.append("סוג פרויקט: " + C.LABELS_PROJECT.get(collected.get("projectType"), ""))
    lines += [
        "",
        "שם מלא: " + collected.get("fullName", ""),
        "עיר: " + collected.get("city", ""),
        "טלפון: " + collected.get("contactPhone", ""),
        "",
        "האם הפרטים נכונים?",
        "",
        "1. כן, אפשר להעביר לנציג",
        "2. להתחיל מחדש",
        "3. לעבור לנציג אנושי",
        "",
        "נא להשיב במספר האפשרות בלבד.",
    ]
    return "\n".join(lines)


def _enter_step(state: dict, step: str) -> str:
    """Move to `step`, reset the invalid counter, and return the prompt to send."""
    state["currentStep"] = step
    state["invalidAttempts"] = 0
    if step == C.CONFIRM_DETAILS:
        return build_summary(state["collectedData"])
    return C.STEP_PROMPT.get(step, "")


def _handoff(state: dict, reason: str, message: str, *, airtable="sync") -> dict:
    state["conversationMode"] = C.MODE_WAITING_FOR_AGENT
    state["currentStep"] = C.WAITING_FOR_AGENT
    state["botEnabled"] = False
    state["handoffReason"] = reason
    complete = reason == C.HANDOFF_READY
    return _result(
        state, message,
        airtable="complete" if complete else airtable,
        notify_owner=complete,
        terminal=True,
    )


def _invalid(state: dict, reprompt: str) -> dict:
    """Register an invalid answer; re-ask, or hand off after MAX_INVALID_ATTEMPTS."""
    state["invalidAttempts"] = state.get("invalidAttempts", 0) + 1
    if state["invalidAttempts"] >= C.MAX_INVALID_ATTEMPTS:
        return _handoff(state, C.HANDOFF_INVALID_RESPONSES, C.MSG_INVALID_HANDOFF)
    return _result(state, reprompt)


def _current_prompt(state: dict) -> str:
    step = state.get("currentStep")
    if step == C.CONFIRM_DETAILS:
        return build_summary(state["collectedData"])
    return C.STEP_PROMPT.get(step, "")


def _route_missing_or_confirm(state: dict) -> dict:
    """After contact details: ask the first missing field, or show the summary."""
    cd = state["collectedData"]
    if not cd.get("fullName"):
        return _result(state, _enter_step(state, C.ASK_MISSING_NAME), airtable="sync")
    if not cd.get("city"):
        return _result(state, _enter_step(state, C.ASK_MISSING_CITY), airtable="sync")
    if not cd.get("contactPhone"):
        return _result(state, _enter_step(state, C.ASK_MISSING_PHONE), airtable="sync")
    return _result(state, _enter_step(state, C.CONFIRM_DETAILS), airtable="sync")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def process(state: Optional[dict], text: str, is_media: bool = False) -> dict:
    """Advance the conversation by one customer message. Pure function.

    Returns {"state": <new state>, "replies": [<messages>], "intents": {...}}.
    """
    state = dict(state) if state else new_state()
    state["collectedData"] = dict(state.get("collectedData") or {})
    state["savedQuestions"] = list(state.get("savedQuestions") or [])
    step = state.get("currentStep", C.START)

    # Bot already silent (handed off / human active) → no reply.
    if not state.get("botEnabled", True) or state.get("conversationMode") in (
        C.MODE_WAITING_FOR_AGENT, C.MODE_HUMAN_ACTIVE
    ):
        return _result(state, [])

    text = text or ""

    # ── Global: explicit request for a human agent (checked on every message) ──
    if not is_media and is_agent_request(text):
        return _handoff(state, C.HANDOFF_CUSTOMER_REQUESTED, C.MSG_AGENT_HANDOFF)

    # ── First contact: send the opening menu, do not treat text as an answer ──
    if step in (C.START, None):
        prompt = _enter_step(state, C.ASK_INQUIRY_TYPE)
        state["conversationMode"] = C.MODE_BOT_COLLECTING
        return _result(state, prompt, airtable="sync")

    is_menu = step in _MENU

    # ── Media during collection: acknowledge, re-ask current step, no advance ──
    if is_media:
        return _result(state, C.MSG_MEDIA_PREFIX + _current_prompt(state))

    # ── On menu steps only: price question / other off-topic question ─────────
    if is_menu:
        if is_price_question(text):
            return _result(state, [C.MSG_PRICE, _current_prompt(state)])
        if "?" in text and _match_menu(step, text) is None:
            state["savedQuestions"].append(text.strip())
            return _result(state, C.MSG_SAVED_QUESTION_PREFIX + _current_prompt(state))

    # ── Menu steps (inquiry type / frame / quantity / project) ────────────────
    if is_menu:
        value = _match_menu(step, text)
        if value is None:
            return _invalid(state, _current_prompt(state))

        cd = state["collectedData"]
        if step == C.ASK_INQUIRY_TYPE:
            cd["inquiryType"] = value
            nxt = {
                "ENTRANCE_DOOR":  C.ASK_ENTRANCE_FRAME,
                "INTERIOR_DOORS": C.ASK_INTERIOR_QUANTITY,
                "MAMAD_DOOR":     C.ASK_PROJECT_TYPE,
                "OTHER":          C.ASK_OTHER_DETAILS,
            }[value]
            return _result(state, _enter_step(state, nxt), airtable="sync")
        if step == C.ASK_ENTRANCE_FRAME:
            cd["frameType"] = value
            return _result(state, _enter_step(state, C.ASK_PROJECT_TYPE), airtable="sync")
        if step == C.ASK_INTERIOR_QUANTITY:
            cd["quantityRange"] = value
            return _result(state, _enter_step(state, C.ASK_PROJECT_TYPE), airtable="sync")
        if step == C.ASK_PROJECT_TYPE:
            cd["projectType"] = value
            return _result(state, _enter_step(state, C.ASK_CONTACT_DETAILS), airtable="sync")

    # ── Free-text: "other" details ────────────────────────────────────────────
    if step == C.ASK_OTHER_DETAILS:
        if not text.strip():
            return _result(state, C.MSG_OTHER_DETAILS)  # empty — re-ask, no penalty
        state["collectedData"]["otherInquiry"] = text.strip()
        return _result(state, _enter_step(state, C.ASK_PROJECT_TYPE), airtable="sync")

    # ── Contact details (single labeled message) ──────────────────────────────
    if step == C.ASK_CONTACT_DETAILS:
        parsed = parse_contact(text)
        cd = state["collectedData"]
        name = validate_name(parsed.get("fullName"))
        city = validate_city(parsed.get("city"))
        phone = normalize_phone(parsed.get("contactPhone"))
        if name:
            cd["fullName"] = name
        if city:
            cd["city"] = city
        if phone:
            cd["contactPhone"] = phone
        return _route_missing_or_confirm(state)

    # ── Missing single fields ─────────────────────────────────────────────────
    if step == C.ASK_MISSING_NAME:
        name = validate_name(text)
        if not name:
            return _invalid(state, C.MSG_MISSING_NAME)
        state["collectedData"]["fullName"] = name
        return _route_missing_or_confirm(state)
    if step == C.ASK_MISSING_CITY:
        city = validate_city(text)
        if not city:
            return _invalid(state, C.MSG_MISSING_CITY)
        state["collectedData"]["city"] = city
        return _route_missing_or_confirm(state)
    if step == C.ASK_MISSING_PHONE:
        phone = normalize_phone(text)
        if not phone:
            return _invalid(state, C.MSG_MISSING_PHONE)
        state["collectedData"]["contactPhone"] = phone
        return _route_missing_or_confirm(state)

    # ── Confirmation ──────────────────────────────────────────────────────────
    if step == C.CONFIRM_DETAILS:
        choice = _match_confirm(text)
        if choice is None:
            return _invalid(state, build_summary(state["collectedData"]))
        if choice == "CONFIRM":
            return _handoff(state, C.HANDOFF_READY, C.MSG_READY)
        if choice == "AGENT":
            return _handoff(state, C.HANDOFF_CUSTOMER_REQUESTED, C.MSG_AGENT_HANDOFF)
        # RESTART — reset only the current inquiry's data, keep the customer.
        state["collectedData"] = {}
        state["savedQuestions"] = []
        return _result(state, _enter_step(state, C.ASK_INQUIRY_TYPE), airtable="sync")

    # ── Unknown step (should never happen) — re-ask safely ────────────────────
    return _result(state, _current_prompt(state) or C.MSG_OPENING)
