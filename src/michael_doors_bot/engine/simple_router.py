"""
simple_router.py — State-machine-based conversation engine.

Architecture:
  Python decides WHAT to ask next (via _decide_next_action).
  Claude decides HOW to phrase it (using DECIDED ACTION block in system prompt).

Pipeline (per incoming message):
  1.  Extract fields from customer message (regex layer)
  2.  Detect new topics from message
  3.  Merge extracted fields + topics into state
  4.  Apply buffered style to current topic (_apply_style_to_topic)
  5.  Advance stage flags (_advance_stage) — reads history, updates state
  6.  Decide next action (_decide_next_action) — pure state function
  7.  Build system prompt with DECIDED ACTION block injected at the end
  8.  Call AI (OpenRouter/GPT-4.1-mini primary, Claude fallback)
  9.  Parse Claude's JSON response (extracted_* fields)
  10. Merge Claude's extracted fields into state
  11. Re-advance stage flags (catch new history entries)
  12. Save state + conversations to disk + Supabase
  13. Return result dict
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .messages import (
    PITCH,
    CONTACT_OPENER,
    STAGE3_QUESTION,
    FINAL_HANDOFF,
    FINAL_HANDOFF_FEMALE,
    FINAL_HANDOFF_MALE,
    FINAL_HANDOFF_SERVICE,
    FINAL_HANDOFF_SERVICE_FEMALE,
    FINAL_HANDOFF_SERVICE_MALE,
    QUESTION_TEMPLATES,
    ERROR_MSG as _ERR,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent.parent.parent
_PROMPT_PATH = _ROOT / "src" / "prompts" / "systemPrompt.txt"
_FAQ_PATH    = _ROOT / "src" / "data" / "faqBank.json"

from .. import config as _cfg  # noqa: E402
_DATA_DIR       = Path(_cfg.DATA_DIR) if _cfg.DATA_DIR else _ROOT
_CONV_PATH      = _DATA_DIR / "conversations.json"
_LAST_SEEN_PATH = _DATA_DIR / "last_seen.json"

_SESSION_GAP = 24 * 3600  # seconds before treating customer as new

# ── System prompt ─────────────────────────────────────────────────────────────
def _load_system_prompt_sync() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.critical("[BOOT] FATAL: Failed to load system prompt: %s", e)
        return ""

_SYSTEM_PROMPT: str = _load_system_prompt_sync()
logger.info("[BOOT] System prompt loaded (%d chars)", len(_SYSTEM_PROMPT))


async def _refresh_system_prompt() -> None:
    global _SYSTEM_PROMPT
    fresh = _load_system_prompt_sync()
    if fresh:
        _SYSTEM_PROMPT = fresh
        DIAG_STATE["system_prompt_loaded"] = True
        DIAG_STATE["system_prompt_chars"] = len(fresh)
        logger.info("[RELOAD] System prompt reloaded (%d chars)", len(fresh))
        try:
            from ..providers.supabase_store import save_system_prompt
            ok = await save_system_prompt(fresh)
            if ok:
                logger.info("[SUPABASE] System prompt synced")
        except Exception as e:
            logger.warning("[SUPABASE] Could not sync system prompt: %s", e)
    else:
        logger.warning("[RELOAD] System prompt file was empty — keeping previous")

# ── FAQ bank ──────────────────────────────────────────────────────────────────
def _load_faq_sync() -> list:
    try:
        return json.loads(_FAQ_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("[BOOT] Failed to load FAQ bank: %s", e)
        return []

_faq_bank: list[dict] = _load_faq_sync()
logger.info("[BOOT] FAQ bank loaded (%d entries)", len(_faq_bank))


async def _refresh_faq() -> None:
    global _faq_bank
    fresh_file = _load_faq_sync()
    if fresh_file:
        _faq_bank = fresh_file
        DIAG_STATE["faq_count"] = len(fresh_file)
        logger.info("[RELOAD] FAQ bank reloaded (%d entries)", len(fresh_file))
    try:
        from ..providers.supabase_store import load_faq
        entries = await load_faq()
        if entries:
            _faq_bank = entries
            DIAG_STATE["faq_count"] = len(entries)
            logger.info("[SUPABASE] FAQ bank refreshed (%d entries)", len(entries))
    except Exception as e:
        logger.warning("[SUPABASE] Could not refresh FAQ: %s", e)


def _check_content_consistency() -> list[str]:
    issues: list[str] = []
    if not _SYSTEM_PROMPT or not _faq_bank:
        return issues
    prompt_phones = set(re.findall(r'0\d{2}-\d{7}', _SYSTEM_PROMPT))
    faq_phones: set[str] = set()
    for entry in _faq_bank:
        faq_phones.update(re.findall(r'0\d{2}-\d{7}', entry.get("answer", "")))
    conflict = faq_phones - prompt_phones
    if conflict:
        issues.append(f"Phone mismatch — FAQ has {conflict}, prompt has {prompt_phones}")
    prompt_addr = set(re.findall(r'בעלי המלאכה\s+\d+', _SYSTEM_PROMPT))
    faq_addr: set[str] = set()
    for entry in _faq_bank:
        faq_addr.update(re.findall(r'בעלי המלאכה\s+\d+', entry.get("answer", "")))
    if faq_addr and prompt_addr and faq_addr != prompt_addr:
        issues.append(f"Address mismatch — FAQ: {faq_addr}, prompt: {prompt_addr}")
    return issues

_consistency_issues = _check_content_consistency()
for _issue in _consistency_issues:
    logger.critical("[CONSISTENCY] %s", _issue)
if not _consistency_issues and _faq_bank:
    logger.info("[BOOT] Content consistency check passed (%d FAQ entries)", len(_faq_bank))

# ── Diagnostics ───────────────────────────────────────────────────────────────
DIAG_STATE: dict = {
    "system_prompt_loaded":  bool(_SYSTEM_PROMPT),
    "system_prompt_chars":   len(_SYSTEM_PROMPT),
    "faq_count":             len(_faq_bank),
    "data_dir":              str(_DATA_DIR),
    "consistency_issues":    _consistency_issues,
    "ai_primary":            f"openrouter/openai/gpt-4.1-mini" if _cfg.OPENROUTER_API_KEY else "claude/claude-sonnet-4-6",
    "ai_fallback":           "claude/claude-sonnet-4-6" if _cfg.OPENROUTER_API_KEY else "none",
    "openrouter_key_set":    bool(_cfg.OPENROUTER_API_KEY),
    "last_ai_provider":      None,
    "openrouter_failures":   0,
    "last_openrouter_error": None,
}

_MAX_INPUT_CHARS = 2000

# ── Conversation history ──────────────────────────────────────────────────────
_conversations: dict[str, list[dict]] = {}
try:
    _conversations = json.loads(_CONV_PATH.read_text(encoding="utf-8"))
except Exception:
    pass


def _save_conversations() -> None:
    try:
        _CONV_PATH.write_text(json.dumps(_conversations), encoding="utf-8")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATION STATE — NEW SCHEMA (v2 — state-machine based)
# ══════════════════════════════════════════════════════════════════════════════

_conv_state: dict[str, dict] = {}
_STATE_PATH = _DATA_DIR / "conv_state.json"

try:
    _conv_state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
except Exception:
    pass


def _save_conv_state() -> None:
    try:
        _STATE_PATH.write_text(
            json.dumps(_conv_state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _empty_conv_state() -> dict:
    """Return a fresh v2 conversation state dict."""
    return {
        # ── Stage flags ──
        "stage3_done":         False,  # True after Stage3 q sent AND customer replied
        "stage4_opener_sent":  False,  # True after contact-opener message sent
        "summary_sent":        False,  # True after Stage5 summary sent

        # ── Topic tracking ──
        "active_topics":       [],     # append-only list of detected topics
        "current_active_topic": None,  # first incomplete topic in priority order

        # ── Entrance door fields ──
        "entrance_scope":        None,   # "with_frame" | "door_only"
        "entrance_style":        None,   # "flat" | "designed" | "undecided"
        "entrance_catalog_sent": False,
        "entrance_model":        None,   # model name | "undecided"

        # ── Interior door fields ──
        "interior_project_type": None,   # "new" | "renovation" | "replacement"
        "interior_quantity":     None,   # int
        "interior_style":        None,   # "flat" | "designed" | "undecided"
        "interior_catalog_sent": False,
        "interior_model":        None,

        # ── Mamad fields ──
        "mamad_type":  None,   # "new" | "replacement"
        # mamad_scope intentionally removed — mamad never asks scope/frame question

        # ── Showroom ──
        "showroom_requested": False,

        # ── Style buffer ──
        "_raw_style": None,  # temporary until topic is known

        # ── Contact fields ──
        "full_name":             None,
        "phone":                 None,
        "city":                  None,
        "preferred_contact_hours": None,

        # ── Customer metadata ──
        "customer_gender_locked": None,  # None | "female" | "male"
        "service_type":           None,
        "referral_source":        None,
        "is_returning_customer":  None,

        # ── Schema version (for migration guard) ──
        "_v": 2,
    }


def _is_v2_state(state: dict) -> bool:
    return state.get("_v") == 2


# ── Topic priority order ──────────────────────────────────────────────────────
_TOPIC_PRIORITY = ["entrance_doors", "interior_doors", "mamad", "showroom_meeting", "repair"]

# Topics that result in a price quote (vs. service / showroom visit)
_PURCHASE_TOPICS: frozenset[str] = frozenset({"entrance_doors", "interior_doors", "mamad"})


def _get_farewell_text(state: dict) -> str:
    """
    Return the correct farewell string based on active topics and customer gender.
    - Purchase topics (entrance/interior/mamad) → "הצעת מחיר מסודרת"
    - Service/info topics (repair/showroom)     → "כל הפרטים"
    """
    gender = state.get("customer_gender_locked")
    active = set(state.get("active_topics") or [])
    is_purchase = bool(active & _PURCHASE_TOPICS)

    if is_purchase:
        if gender == "female": return FINAL_HANDOFF_FEMALE
        if gender == "male":   return FINAL_HANDOFF_MALE
        return FINAL_HANDOFF
    else:
        if gender == "female": return FINAL_HANDOFF_SERVICE_FEMALE
        if gender == "male":   return FINAL_HANDOFF_SERVICE_MALE
        return FINAL_HANDOFF_SERVICE


# ── Topic → natural Hebrew label ─────────────────────────────────────────────
# Used in Stage 5 summary so customers never see internal field names like
# "['interior_doors', 'entrance_doors']".
_TOPIC_LABELS_HE: dict[str, str] = {
    "entrance_doors":   "דלת כניסה",
    "interior_doors":   "דלתות פנים",
    "mamad":            'דלת ממ"ד',
    "showroom_meeting": "ביקור באולם תצוגה",
    "repair":           "תיקון דלת",
}


def _topic_label_he(state: dict) -> str:
    """
    Return a natural Hebrew description of what the customer needs.
    Priority: service_type field (Claude-extracted free text) → mapped active_topics → fallback.
    Multiple topics are joined with ' + '.
    Never exposes internal field names.
    """
    service = state.get("service_type")
    if service:
        return service
    active = state.get("active_topics") or []
    parts = [_TOPIC_LABELS_HE.get(t, t) for t in active]
    return " + ".join(parts) if parts else "שירות דלתות"


# ── Hebrew style / project-type labels for rich summary ──────────────────────
_STYLE_HE: dict[str, str] = {
    "flat":     "חלקות",
    "designed": "מעוצבות",
}
_PROJ_HE: dict[str, str] = {
    "new":         "בית חדש",
    "renovation":  "שיפוץ",
    "replacement": "החלפה",
}


def _build_service_label_he(state: dict) -> str:
    """
    Build a natural Hebrew service label for Stage 5 summary.
    Derived entirely from collected state fields — never exposes internal topic keys
    or English field names (entrance_doors, interior_doors, etc.).

    Examples:
      entrance_doors only                    → "דלת כניסה"
      interior_doors qty=3 style=flat        → "3 דלתות פנים חלקות"
      interior_doors qty=3 style=flat proj=renovation → "3 דלתות פנים חלקות — שיפוץ"
      entrance_doors + interior_doors        → "דלת כניסה + 3 דלתות פנים חלקות"
    """
    active = state.get("active_topics") or []
    parts: list[str] = []

    for topic in active:
        if topic == "entrance_doors":
            parts.append("דלת כניסה")

        elif topic == "interior_doors":
            qty   = state.get("interior_quantity")
            style = state.get("interior_style")
            proj  = state.get("interior_project_type")
            qty_str   = f"{qty} " if qty else ""
            style_str = f" {_STYLE_HE[style]}" if style in _STYLE_HE else ""
            proj_str  = f" — {_PROJ_HE[proj]}" if proj in _PROJ_HE else ""
            parts.append(f"{qty_str}דלתות פנים{style_str}{proj_str}")

        elif topic == "mamad":
            mamad_type = state.get("mamad_type")
            if mamad_type == "new":
                parts.append('דלת ממ"ד חדשה')
            elif mamad_type == "replacement":
                parts.append('החלפת דלת ממ"ד')
            else:
                parts.append('דלת ממ"ד')

        elif topic == "showroom_meeting":
            parts.append("ביקור באולם תצוגה")

        elif topic == "repair":
            repair_type = state.get("repair_type")
            if repair_type == "entrance":
                parts.append("תיקון דלת כניסה")
            elif repair_type == "interior":
                parts.append("תיקון דלת פנים")
            else:
                parts.append("תיקון דלת")

        else:
            # Unknown topic: map via _TOPIC_LABELS_HE, fall back to raw key
            parts.append(_TOPIC_LABELS_HE.get(topic, topic))

    return " + ".join(parts) if parts else "שירות דלתות"


# ══════════════════════════════════════════════════════════════════════════════
# FIELD EXTRACTION — REGEX LAYER
# ══════════════════════════════════════════════════════════════════════════════

_ISRAELI_CITIES: set[str] = {
    "נתיבות", "באר שבע", "אשקלון", "אשדוד", "אופקים", "שדרות", "רהט", "דימונה",
    "קריית גת", "קריית מלאכי", "ערד", "אילת", "מצפה רמון", "ירוחם", "עומר",
    "להבים", "מיתר", "כסייפה", "חורה", "תל שבע", "לקיה",
    "תל אביב", "ירושלים", "חיפה", "ראשון לציון", "פתח תקווה", "נתניה",
    "בני ברק", "חולון", "רמת גן", "מודיעין", "כפר סבא", "הרצליה",
    "רחובות", "בת ים", "בית שמש", "עפולה", "נהריה", "טבריה", "לוד",
    "רמלה", "נצרת", "רעננה", "הוד השרון", "קריית אונו", "אור יהודה",
    "מזכרת בתיה", "גדרה", "יבנה", "גן יבנה", "ראש העין", "כפר יונה",
    "טירת כרמל", "עכו", "כרמיאל", "צפת", "קריית ביאליק", "קריית מוצקין",
    "קריית ים", "קריית אתא", "מגדל העמק", "זכרון יעקב", "חדרה",
    "אום אל פחם", "שפרעם", "גבעתיים", "אריאל", "מעלה אדומים",
    "מודיעין עילית", "ביתר עילית", "בית שאן", "יוקנעם", "קצרין",
    "אלעד", "גבעת שמואל", "אור עקיבא", "נס ציונה", "גבעת ברנר",
    'ב"ש', 'ת"א',
}

# ── Hebrew-only enforcement ───────────────────────────────────────────────────
# Any message containing at least one Hebrew letter is treated as Hebrew.
# The fallback fires ONLY for messages that contain foreign-language letters
# (Latin/Cyrillic/Arabic) with zero Hebrew characters AND zero digits.
# Numeric-only ("3"), phone numbers, emojis, and punctuation all pass through.
_HEB_CHAR_RE     = re.compile(r'[\u05d0-\u05fa]')
_FOREIGN_LETTER_RE = re.compile(r'[a-zA-Z\u0400-\u04FF\u0600-\u06FF]')
_HEBREW_ONLY_REPLY = "כרגע אני יכולה לעזור בעברית 😊 אפשר לכתוב לי בעברית במה מדובר?"


def _has_hebrew(text: str) -> bool:
    """Return True if the text contains at least one Hebrew character."""
    return bool(_HEB_CHAR_RE.search(text))


def _needs_hebrew_fallback(text: str) -> bool:
    """
    Return True only when the message should get the Hebrew-only fallback.

    Rules (in priority order):
    1. Has Hebrew chars → False  (normal flow)
    2. Has digits       → False  (numeric answer, phone number, quantity)
    3. Has foreign-language letters (Latin/Cyrillic/Arabic) → True  (fallback)
    4. Otherwise (emoji, punctuation, empty) → False (normal flow)
    """
    stripped = text.strip()
    if not stripped:
        return False
    if _HEB_CHAR_RE.search(stripped):
        return False
    if re.search(r'\d', stripped):
        return False
    return bool(_FOREIGN_LETTER_RE.search(stripped))


_PHONE_RE = re.compile(
    r'(?<!\d)'
    r'(\+?972[-\s]?|0)'
    r'([5][0-9][-\s]?[0-9]{3}[-\s]?[0-9]{4}'
    r'|[5][0-9]{8})'
    r'(?!\d)'
)

# Near-miss: looks like a phone (starts with 05) but has only 8–9 digits total
# (missing 1–2 digits). Does NOT overlap with valid 10-digit numbers.
_NEAR_MISS_PHONE_RE = re.compile(r'(?<!\d)(0[5][0-9]{6,7})(?!\d)')

_HEB_WORD_RE = re.compile(r'^[\u05d0-\u05fa][\u05d0-\u05fa\'\- ]{0,35}$')

# Common single Hebrew words that are never a person's name
_NOT_A_NAME: frozenset[str] = frozenset({
    'כן', 'לא', 'אולי', 'טוב', 'בסדר', 'אחלה', 'נהדר', 'מעולה', 'סבבה',
    'ברור', 'אוקי', 'נכון', 'חלקה', 'חלקות', 'מעוצבת', 'מעוצבות',
    'חדשה', 'חדש', 'שיפוץ', 'החלפה', 'תיקון', 'פנים', 'כניסה',
})

# ── Topic patterns ────────────────────────────────────────────────────────────
_TOPIC_PATTERNS: dict[str, re.Pattern] = {
    "entrance_doors": re.compile(
        r"דלת כניסה|דלתות כניסה"
        r"|דלת חוץ|דלתות חוץ"
        r"|דלת חיצונית|דלתות חיצוניות"
        r"|דלת ראשית|דלתות ראשיות"
        r"|דלת ברזל|דלת פלדה|דלתות ברזל|דלתות פלדה"
        r"|כניסה לבית|כניסה לדירה|כניסה לבניין"
        r"|נפחות|נפחת|פנורמי|יווני|מרקורי|עדן|קלאסי|אומנויות|סביליה",
        re.IGNORECASE,
    ),
    "interior_doors": re.compile(
        r"דלת פנים|דלתות פנים"
        r"|דלת לחדר|דלת חדר|דלתות חדר"
        r"|דלת שינה|דלת שירותים|דלת אמבטיה|דלת מטבח|דלת סלון"
        r"|דלתות פנימיות|פולימר",
        re.IGNORECASE,
    ),
    "mamad": re.compile(
        r'ממ"ד|ממד|מרחב מוגן|חדר ביטחון|דלת ממד',
        re.IGNORECASE,
    ),
    "showroom_meeting": re.compile(
        r"לבוא לאולם|תיאום ביקור|לראות מקרוב|מתי אפשר לבוא"
        r"|לקבוע פגישה|לבוא לחנות|רוצה להגיע|אפשר לקבוע פגישה"
        r"|אולם תצוגה|אולם התצוגה"
        r"|איפה אתם נמצאים|היכן אתם נמצאים|איפה אתם|היכן אתם"
        r"|יש לכם אולם|יש אולם|יש לכם חנות|יש לכם מקום"
        r"|אפשר להגיע לאולם|לראות דגמים|לבוא לראות",
        re.IGNORECASE,
    ),
    "repair": re.compile(
        r"תיקון|תקלה|בעיה בדלת|הדלת לא נסגרת|הדלת לא נפתחת"
        r"|הדלת תקועה|ציר שבור|מנעול שבור|ידית שבורה"
        r"|פריצה|פרצו|שוד|חירום|עזרה דחופה"
        r"|התפרקה|נשברה|שירות לדלת",
        re.IGNORECASE,
    ),
}


def _detect_topics_from_message(msg: str) -> list[str]:
    """Return list of topics detected in a message."""
    return [topic for topic, pat in _TOPIC_PATTERNS.items() if pat.search(msg)]


def _extract_fields_from_message(text: str, state: dict | None = None) -> dict:
    """
    Regex extraction of structured fields from a customer message.
    Returns only fields that were confidently found.
    Uses state.current_active_topic to route style answers to the correct field.
    """
    extracted: dict = {}
    t = text.strip()
    current_topic = (state or {}).get("current_active_topic")

    # ── Phone ─────────────────────────────────────────────────────────────────
    phone_match = _PHONE_RE.search(t)
    if phone_match:
        raw = re.sub(r'[-\s+]', '', phone_match.group(0))
        if raw.startswith('972'):
            raw = '0' + raw[3:]
        elif raw.startswith('+972'):
            raw = '0' + raw[4:]
        extracted['phone'] = raw
    else:
        # No valid phone found — check for near-miss (looks like a phone but too short)
        nm = _NEAR_MISS_PHONE_RE.search(t)
        if nm:
            extracted['_near_miss_phone'] = nm.group(0)

    # ── City ──────────────────────────────────────────────────────────────────
    for city in _ISRAELI_CITIES:
        if city in t:
            extracted['city'] = city
            break
    if 'city' not in extracted:
        city_prep = re.search(
            r'(?:מ|ב|ל|ו)(נתיבות|באר שבע|אשקלון|אשדוד|אופקים|שדרות'
            r'|ירושלים|תל אביב|חיפה|ראשון לציון|פתח תקווה|נתניה|רחובות)',
            t)
        if city_prep:
            extracted['city'] = city_prep.group(1)

    # ── Name ──────────────────────────────────────────────────────────────────
    # Strategy: when a phone number is present, remove it and any detected city
    # from the message, then treat the remainder as the name candidate.
    # This handles all orderings and comma/space separators:
    #   "ליטל 0523989366"
    #   "ליטל, אשקלון, 0523989366"
    #   "0523989366 ליטל אשקלון"
    #   "שמי דוד כהן, 052-1234567, תל אביב"
    if phone_match:
        # Build remainder: everything except the phone number
        remainder = (t[:phone_match.start()] + ' ' + t[phone_match.end():]).strip()

        # Remove detected city from remainder
        if 'city' in extracted:
            remainder = remainder.replace(extracted['city'], '')

        # Remove common name-introduction prefixes
        remainder = re.sub(
            r'^(?:שמי|קוראים לי|אני|שם שלי|השם שלי)\s*', '',
            remainder, flags=re.IGNORECASE,
        )

        # Normalize: commas and punctuation → spaces, collapse whitespace
        remainder = re.sub(r'[,،.!?;]+', ' ', remainder)
        remainder = re.sub(r'\s+', ' ', remainder).strip()

        if (remainder
                and _HEB_WORD_RE.match(remainder)
                and remainder not in _ISRAELI_CITIES
                and remainder not in _NOT_A_NAME
                and len(remainder) >= 2):
            extracted['full_name'] = remainder
    else:
        # No phone in this message — try two strategies:

        # 1) Explicit name-introduction markers (always)
        name_m = re.match(
            r'^(?:שמי|קוראים לי|שם שלי|אני)\s+([\u05d0-\u05fa][\u05d0-\u05fa\'\- ]{1,30})',
            t, re.IGNORECASE,
        )
        if name_m:
            candidate = name_m.group(1).strip()
            if candidate not in _ISRAELI_CITIES:
                extracted['full_name'] = candidate

        # 2) Loose match: when phone is already in state and name is still missing,
        #    a short Hebrew-only message is very likely just the customer's name.
        #    Accepts first name alone (e.g. "ליטל" or "דוד כהן").
        elif (
            not extracted.get('full_name')
            and (state or {}).get('phone')
            and not (state or {}).get('full_name')
        ):
            plain = re.match(r'^([\u05d0-\u05fa][\u05d0-\u05fa\'\- ]{1,30})$', t.strip())
            if plain:
                candidate = plain.group(1).strip()
                if (candidate not in _ISRAELI_CITIES
                        and candidate not in _NOT_A_NAME
                        and len(candidate) >= 2):
                    extracted['full_name'] = candidate

    # ── Gender ────────────────────────────────────────────────────────────────
    if re.search(r'מחפשת|צריכה\b|מתעניינת|שמחה\b|מרוצה\b|מעוניינת|רציתי|קניתי\b|הגעתי\b', t):
        extracted['customer_gender_locked'] = 'female'
    elif re.search(r'מחפש\b|צריך\b|מתעניין\b|שמח\b|מעוניין\b', t):
        extracted['customer_gender_locked'] = 'male'

    # ── Entrance scope (entrance_doors only — never for mamad) ───────────────
    # mamad never asks "כולל משקוף" — scope is irrelevant for mamad pricing.
    if current_topic != "mamad":
        if re.search(r'כולל משקוף|עם משקוף|דלת ומשקוף', t, re.IGNORECASE):
            extracted['entrance_scope'] = "with_frame"
        elif re.search(r'דלת בלבד|בלי משקוף|רק דלת\b|ללא משקוף|דלת לבד', t, re.IGNORECASE):
            extracted['entrance_scope'] = "door_only"

    # ── Style ─────────────────────────────────────────────────────────────────
    # Route to topic-specific field based on current active topic; buffer if unknown
    if re.search(r'\bחלקה\b|\bחלקות\b', t, re.IGNORECASE):
        style_val = "flat"
        if current_topic == "entrance_doors":
            extracted['entrance_style'] = style_val
        elif current_topic == "interior_doors":
            extracted['interior_style'] = style_val
        else:
            extracted['_raw_style'] = style_val
    elif re.search(r'\bמעוצבת\b|\bמעוצבות\b', t, re.IGNORECASE):
        style_val = "designed"
        if current_topic == "entrance_doors":
            extracted['entrance_style'] = style_val
        elif current_topic == "interior_doors":
            extracted['interior_style'] = style_val
        else:
            extracted['_raw_style'] = style_val

    # ── Interior project type ─────────────────────────────────────────────────
    if re.search(r'בית חדש|דירה חדשה|נכס חדש', t, re.IGNORECASE):
        extracted['interior_project_type'] = 'new'
    elif re.search(r'\bשיפוץ\b|בשיפוץ\b|משפצים', t, re.IGNORECASE):
        extracted['interior_project_type'] = 'renovation'
    elif re.search(r'\bהחלפה\b|להחליף\b|דלת ישנה|קיימות', t, re.IGNORECASE):
        extracted['interior_project_type'] = 'replacement'

    # ── Mamad type ────────────────────────────────────────────────────────────
    if re.search(r'ממ.?ד חדש|מרחב מוגן חדש', t, re.IGNORECASE):
        extracted['mamad_type'] = 'new'
    elif re.search(r'ממ.?ד קיים|להחליף.*ממ.?ד|ממ.?ד.*להחליף|החלפת.*ממ.?ד|החלפה.*ממ.?ד', t, re.IGNORECASE):
        extracted['mamad_type'] = 'replacement'

    # ── Interior quantity ─────────────────────────────────────────────────────
    count_m = re.search(r'(\d+)\s*דלתות', t)
    if count_m:
        extracted['interior_quantity'] = int(count_m.group(1))
    else:
        # Hebrew number words (e.g. "שלוש דלתות פנים")
        _HEB_QTY: dict[str, int] = {
            'אחת': 1, 'אחד': 1,
            'שניים': 2, 'שתיים': 2, 'שני': 2, 'שתי': 2,
            'שלוש': 3, 'שלושה': 3,
            'ארבע': 4, 'ארבעה': 4,
            'חמש': 5, 'חמישה': 5,
            'שש': 6, 'ששה': 6,
            'שבע': 7, 'שבעה': 7,
            'שמונה': 8,
            'תשע': 9, 'תשעה': 9,
            'עשר': 10, 'עשרה': 10,
        }
        for word, num in _HEB_QTY.items():
            if re.search(rf'(?:{word})\s*(?:דלתות|דלת)|(?:דלתות|דלת)\s*(?:{word})', t, re.IGNORECASE):
                extracted['interior_quantity'] = num
                break

    # ── Showroom requested ────────────────────────────────────────────────────
    if re.search(
        r'לבוא לאולם|לבקר|לבוא אליכם|לקבוע פגישה|ביקור באולם|מתי אפשר לבוא'
        r'|אפשר להגיע|רוצה להגיע|לראות מקרוב'
        r'|אולם תצוגה|אולם התצוגה|איפה אתם נמצאים|היכן אתם'
        r'|יש לכם אולם|יש אולם|אפשר להגיע לאולם|לראות דגמים|לבוא לראות',
        t, re.IGNORECASE
    ):
        extracted['showroom_requested'] = True

    # ── Preferred contact hours ───────────────────────────────────────────────
    hours_m = re.search(r'אחרי\s*(\d{1,2})', t)
    if hours_m:
        h = int(hours_m.group(1))
        if h < 12:
            h += 12
        extracted['preferred_contact_hours'] = f'אחרי {h:02d}:00'
    elif re.search(r'בכל שעה|בכל זמן|לא משנה', t, re.IGNORECASE):
        extracted['preferred_contact_hours'] = 'בכל שעה'

    return extracted


# ══════════════════════════════════════════════════════════════════════════════
# STATE MERGING
# ══════════════════════════════════════════════════════════════════════════════

def _merge_state(existing: dict, new_fields: dict) -> dict:
    """
    Merge new_fields into existing state dict.
    Rules:
    - Never overwrite a non-null field with null.
    - gender_locked: set once, never changed.
    - active_topics: union (append-only).
    - Boolean flags: only update False→True, never True→False.
    - _new_topics key: appended to active_topics, then dropped.
    """
    merged = dict(existing)

    for key, value in new_fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        if key == '_new_topics':
            existing_list = merged.get('active_topics') or []
            topics = value if isinstance(value, list) else [value]
            for t in topics:
                if t not in existing_list:
                    existing_list.append(t)
            merged['active_topics'] = existing_list

        elif key == '_near_miss_phone':
            # Only store a near-miss when we don't have a valid phone yet
            if not merged.get('phone'):
                merged[key] = value

        elif key == 'customer_gender_locked':
            if not merged.get('customer_gender_locked'):
                merged[key] = value

        elif key in ('stage3_done', 'stage4_opener_sent', 'summary_sent',
                     'entrance_catalog_sent', 'interior_catalog_sent',
                     'showroom_requested'):
            # Boolean flags — only update False→True
            if value and not merged.get(key):
                merged[key] = True

        else:
            # All other fields: only update if currently None
            if merged.get(key) is None:
                merged[key] = value

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# NEXT ACTION DECISION — STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NextAction:
    stage:        int   # 1–7
    field_to_ask: str   # logical field name
    template_key: str   # key into QUESTION_TEMPLATES (or special value)
    is_fixed:     bool  # True = send template verbatim, no rephrasing
    context:      str   # human-readable description for logs / action block


def _topic_complete(topic: str, state: dict) -> bool:
    """Return True when all required fields for the given topic are collected."""
    if topic == "entrance_doors":
        scope = state.get("entrance_scope")
        style = state.get("entrance_style")
        if scope is None or style is None:
            return False
        if style == "flat":
            return True
        # designed/undecided → complete once catalog has been sent.
        # entrance_model is saved passively if the customer mentions one,
        # but it never blocks progression to the next topic.
        return bool(state.get("entrance_catalog_sent"))

    if topic == "interior_doors":
        if state.get("interior_project_type") is None:
            return False
        if state.get("interior_quantity") is None:
            return False
        style = state.get("interior_style")
        if style is None:
            return False
        if style == "flat":
            return True
        # designed/undecided → complete once catalog has been sent.
        # interior_model is saved passively if the customer mentions one.
        return bool(state.get("interior_catalog_sent"))

    if topic == "mamad":
        # mamad is complete once mamad_type is known — no scope question
        return state.get("mamad_type") is not None

    if topic == "showroom_meeting":
        # No product questions for showroom — complete as soon as topic is detected.
        # Stage 3 is asked AFTER contact collection (handled in _decide_next_action).
        return True

    if topic == "repair":
        # Repair has no product fields — always "complete" for queue purposes
        # (goes straight to contact collection)
        return True

    return False


def _compute_current_topic(state: dict) -> str | None:
    """Return the first incomplete topic in priority order (fix 1)."""
    active = state.get("active_topics") or []
    for topic in _TOPIC_PRIORITY:
        if topic in active and not _topic_complete(topic, state):
            return topic
    return None


def _next_topic_action(topic: str, state: dict) -> NextAction | None:
    """Return the next required NextAction for the given topic."""
    if topic == "entrance_doors":
        if state.get("entrance_scope") is None:
            return NextAction(2, "entrance_scope", "ask_entrance_scope", False,
                              "entrance: ask scope (with_frame or door_only)")
        if state.get("entrance_style") is None:
            return NextAction(2, "entrance_style", "ask_entrance_style", False,
                              "entrance: ask style (flat or designed)")
        if state.get("entrance_style") in ("designed", "undecided"):
            if not state.get("entrance_catalog_sent"):
                return NextAction(2, "entrance_catalog", "entrance_catalog", True,
                                  "entrance: send catalog URL (informational — does not block flow)")
            # catalog sent → entrance topic complete; model saved passively if mentioned
        return None  # complete

    if topic == "interior_doors":
        if state.get("interior_project_type") is None:
            return NextAction(2, "interior_project_type", "ask_interior_project_type", False,
                              "interior: ask project type (new/renovation/replacement)")
        if state.get("interior_quantity") is None:
            return NextAction(2, "interior_quantity", "ask_interior_quantity", False,
                              "interior: ask quantity")
        if state.get("interior_style") is None:
            return NextAction(2, "interior_style", "ask_interior_style", False,
                              "interior: ask style (flat or designed)")
        if state.get("interior_style") in ("designed", "undecided"):
            if not state.get("interior_catalog_sent"):
                return NextAction(2, "interior_catalog", "interior_catalog", True,
                                  "interior: send catalog URL (informational — does not block flow)")
            # catalog sent → interior topic complete; model saved passively if mentioned
        return None

    if topic == "mamad":
        # Only one question: new or replacement. No scope/frame question for mamad.
        if state.get("mamad_type") is None:
            return NextAction(2, "mamad_type", "ask_mamad_type", False,
                              "mamad: ask type (new or replacing existing)")
        return None

    if topic == "showroom_meeting":
        # showroom_meeting: just needs the customer to confirm they want to visit
        # (showroom_requested is set by detection); queue will complete once True
        return None

    if topic == "repair":
        # repair: no product questions — skips directly to contact
        return None

    return None


def _get_callback_key(state: dict) -> str:
    active = state.get("active_topics") or []
    is_showroom_only = (set(active) == {"showroom_meeting"})
    gender = state.get("customer_gender_locked")
    if is_showroom_only:
        if gender == "female": return "ask_callback_time_showroom_female"
        if gender == "male":   return "ask_callback_time_showroom_male"
        return "ask_callback_time_showroom_neutral"
    if gender == "female": return "ask_callback_time_female"
    if gender == "male":   return "ask_callback_time_male"
    return "ask_callback_time_neutral"


def _decide_next_action(state: dict) -> NextAction:
    """
    Pure state machine — decide the next action based solely on conversation state.
    Called after _advance_stage() has updated all flags.
    Fix 3: always returns something (safe fallback if nothing matched).
    """
    try:
        active = state.get("active_topics") or []
        current_topic = state.get("current_active_topic")

        # ── Stage 2: topic qualification ──────────────────────────────────────
        if not active:
            # No topics detected yet
            return NextAction(2, "topic_detection", "ask_topic_clarification", False,
                              "no topics detected — ask what type of door they need")

        if current_topic:
            action = _next_topic_action(current_topic, state)
            if action:
                return action

        # ── All topic queues complete ─────────────────────────────────────────

        # repair-only skips Stage 3 entirely
        is_repair_only   = (active == ["repair"])
        # showroom-only skips pre-contact Stage 3; does post-contact Stage 3 instead
        is_showroom_only = (set(active) == {"showroom_meeting"})

        # Stage 3: pre-contact wrap-up (gender-aware)
        # Skip for repair-only AND showroom-only (showroom Stage 3 comes after contacts)
        if not state.get("stage3_done") and not is_repair_only and not is_showroom_only:
            gender = state.get("customer_gender_locked")
            stage3_key = (
                "stage3_question_female" if gender == "female" else
                "stage3_question_male"   if gender == "male"   else
                "stage3_question"
            )
            return NextAction(3, "stage3_question", stage3_key, True,
                              "stage 3: ask if anything else before contact collection")

        # Stage 4: contact opener
        if not state.get("stage4_opener_sent"):
            opener_key = (
                "contact_opener_showroom"
                if "showroom_meeting" in active
                else "contact_opener"
            )
            return NextAction(4, "contact_opener", opener_key, True,
                              "stage 4: send contact-collection opener (no question appended)")

        # Stage 4: collect contact fields one at a time
        if not state.get("phone"):
            return NextAction(4, "phone", "ask_phone", False, "stage 4: ask phone")
        if not state.get("full_name"):
            return NextAction(4, "full_name", "ask_name", False, "stage 4: ask name")
        if not state.get("city"):
            return NextAction(4, "city", "ask_city", False, "stage 4: ask city")

        # Showroom post-contact Stage 3 — after all contacts collected, ask about preferences
        if is_showroom_only and not state.get("stage3_done"):
            gender = state.get("customer_gender_locked")
            showroom_s3_key = (
                "ask_showroom_stage3_female" if gender == "female" else
                "ask_showroom_stage3_male"   if gender == "male"   else
                "ask_showroom_stage3_neutral"
            )
            return NextAction(3, "stage3_question", showroom_s3_key, True,
                              "showroom stage 3: ask about door preferences after contact collection")

        # Stage 5: summary + confirmation
        if not state.get("summary_sent"):
            return NextAction(5, "summary", "_summary_dynamic", False,
                              "stage 5: send summary and ask הכל נכון?")

        # Stage 6: callback time — is_fixed=True so Claude sends the exact template
        if not state.get("preferred_contact_hours"):
            return NextAction(6, "preferred_contact_hours", _get_callback_key(state), True,
                              "stage 6: ask preferred callback time")

        # Stage 7: farewell + handoff
        return NextAction(7, "farewell", "_farewell_dynamic", True,
                          "stage 7: send farewell message, set handoff_to_human=true")

    except Exception as exc:
        logger.error("[DECIDE:ERR] Unexpected error in _decide_next_action: %s", exc)
        # Fix 3: final fallback — always return something safe
        return NextAction(2, "fallback", "ask_safe_fallback", False,
                          "fallback: unexpected state — ask safe clarification")


# ── Stage advancement (reads history, updates state flags) ────────────────────

def _compute_stage3_done_from_history(history: list[dict]) -> bool:
    """True if Stage 3 question has been sent AND customer replied after it.
    Detects all gender/flow variants by matching shared substrings."""
    _STAGE3_MARKERS = (
        "יש עוד משהו נוסף שנוכל",    # standard variants (neutral/female/male)
        "יש עוד משהו ספציפי שחשוב",   # showroom variant
    )
    found = False
    for msg in history:
        if not found and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if any(m in content for m in _STAGE3_MARKERS):
                found = True
        elif found and msg.get("role") == "user":
            return True
    return False


def _advance_stage(state: dict, history: list[dict]) -> None:
    """
    Update all stage flags based on conversation history.
    Called before _decide_next_action() and again after the AI reply is stored.
    """
    # stage3_done
    if not state.get("stage3_done"):
        if _compute_stage3_done_from_history(history):
            state["stage3_done"] = True

    # stage4_opener_sent — check for contact opener in history.
    # "אשמח לשם" is the common prefix in ALL opener variants:
    #   standard:  "אשמח לשם, עיר ומספר טלפון"
    #   showroom:  "אשמח לשם מלא, עיר ומספר טלפון"
    if not state.get("stage4_opener_sent"):
        for m in history:
            if m.get("role") == "assistant" and "אשמח לשם" in m.get("content", ""):
                state["stage4_opener_sent"] = True
                break

    # summary_sent — check for summary marker
    if not state.get("summary_sent"):
        for m in history:
            if m.get("role") == "assistant" and "הכל נכון?" in m.get("content", ""):
                state["summary_sent"] = True
                break

    # entrance_catalog_sent
    if not state.get("entrance_catalog_sent"):
        for m in history:
            if m.get("role") == "assistant" and "catalog/entry-designed" in m.get("content", ""):
                state["entrance_catalog_sent"] = True
                break

    # interior_catalog_sent
    if not state.get("interior_catalog_sent"):
        for m in history:
            if m.get("role") == "assistant" and "catalog/interior-smooth" in m.get("content", ""):
                state["interior_catalog_sent"] = True
                break

    # Update current_active_topic (fix 1: always first incomplete topic by priority)
    state["current_active_topic"] = _compute_current_topic(state)


# ── Style buffer routing ──────────────────────────────────────────────────────

def _apply_style_to_topic(state: dict) -> None:
    """Route buffered _raw_style to the correct topic-specific field."""
    raw = state.get("_raw_style")
    if raw is None:
        return
    current = state.get("current_active_topic")
    if current == "entrance_doors" and state.get("entrance_style") is None:
        state["entrance_style"] = raw
        state["_raw_style"] = None
        logger.info("[STYLE:APPLY] entrance_style = %s (from buffer)", raw)
    elif current == "interior_doors" and state.get("interior_style") is None:
        state["interior_style"] = raw
        state["_raw_style"] = None
        logger.info("[STYLE:APPLY] interior_style = %s (from buffer)", raw)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _state_summary_block(state: dict) -> str:
    """
    Build a concise state summary block injected into every Claude call.
    Shows Claude all collected fields so it never re-asks them.
    """
    def v(val) -> str:
        if val is None:   return "null"
        if val is False:  return "false"
        if val is True:   return "true"
        return str(val)

    lines = [
        "## COLLECTED STATE — DO NOT RE-ASK ANY FIELD MARKED AS SET",
        "",
        f"Contact:   phone={v(state.get('phone'))}  name={v(state.get('full_name'))}  city={v(state.get('city'))}  callback={v(state.get('preferred_contact_hours'))}",
        "",
        # For flat style: catalog is never needed — mark explicitly to prevent Claude hallucinating a URL
        f"Entrance:  scope={v(state.get('entrance_scope'))}  style={v(state.get('entrance_style'))}"
        + ("  catalog=N/A(flat style needs no catalog — DO NOT send any URL)" if state.get('entrance_style') == 'flat' else f"  catalog_sent={v(state.get('entrance_catalog_sent'))}  model={v(state.get('entrance_model'))}"),
        f"Interior:  type={v(state.get('interior_project_type'))}  qty={v(state.get('interior_quantity'))}  style={v(state.get('interior_style'))}"
        + ("  catalog=N/A(flat style needs no catalog — DO NOT send any URL)" if state.get('interior_style') == 'flat' else f"  catalog_sent={v(state.get('interior_catalog_sent'))}  model={v(state.get('interior_model'))}"),
        f"Mamad:     type={v(state.get('mamad_type'))}",
        f"Showroom:  requested={v(state.get('showroom_requested'))}",
        "",
        f"Topics:    {state.get('active_topics', [])}",
        f"Current:   {v(state.get('current_active_topic'))}",
        f"Stage3done={v(state.get('stage3_done'))}  Opener_sent={v(state.get('stage4_opener_sent'))}  Summary_sent={v(state.get('summary_sent'))}",
        "",
    ]
    gender = state.get("customer_gender_locked")
    if gender == "female":
        lines.append("Gender: FEMALE → use לך / אלייך / תוכלי / תשאירי in every reply.")
    elif gender == "male":
        lines.append("Gender: MALE → use לך / אליך / תוכל / תשאיר in every reply.")
    else:
        lines.append("Gender: UNKNOWN → use neutral plural: לכם / אליכם / תוכלו / תשאירו.")
    return "\n".join(lines)


def _peek_catalog_next_action(
    catalog_action: NextAction, state: dict
) -> tuple:
    """
    After sending this catalog and marking it as sent, what is the next required action?
    Returns (next_action, next_topic_key) or (None, None).

    Used to fold the next question into the same reply as the catalog link,
    so the customer never has to send a blank message just to move the flow forward.

    Only folds in Stage-2 topic-queue questions — Stage 3+ has its own handling.
    """
    peek = dict(state)
    if catalog_action.template_key == "entrance_catalog":
        peek["entrance_catalog_sent"] = True
    elif catalog_action.template_key == "interior_catalog":
        peek["interior_catalog_sent"] = True
    else:
        return None, None

    peek["current_active_topic"] = _compute_current_topic(peek)
    nxt = _decide_next_action(peek)

    # Only fold in Stage 2 topic-queue questions (not another catalog, not fallback)
    if (
        nxt.stage == 2
        and nxt.template_key not in (
            "entrance_catalog", "interior_catalog",
            "ask_topic_clarification", "ask_safe_fallback",
        )
    ):
        return nxt, peek.get("current_active_topic")
    return None, None


def _build_action_block(action: NextAction, state: dict, is_first_message: bool) -> str:
    """
    Build the DECIDED ACTION block that tells Claude exactly what to do this turn.
    This is the authoritative directive — Claude must follow it.
    """
    template_text = QUESTION_TEMPLATES.get(action.template_key, "")
    # For special dynamic keys, template_text will be empty — the action block describes what to do

    gender = state.get("customer_gender_locked")
    gender_note = (
        "Gender: FEMALE → לך/אלייך/תוכלי/תשאירי" if gender == "female" else
        "Gender: MALE → לך/אליך/תוכל/תשאיר"     if gender == "male"   else
        "Gender: UNKNOWN → neutral plural: לכם/אליכם/תוכלו"
    )

    lines = [
        "## DECIDED ACTION — MANDATORY (Python state machine decided this)",
        f"Stage={action.stage}  Field={action.field_to_ask}",
        f"Description: {action.context}",
        "",
    ]

    # ── First message ─────────────────────────────────────────────────────────
    if is_first_message:
        lines += [
            "FIRST MESSAGE RULES:",
            f'  reply_text:   company pitch — EXACT TEXT: {PITCH!r}',
            "  reply_text_2: your actual response to the customer's message.",
            "",
        ]
        if action.is_fixed and template_text:
            lines += [
                f"  For reply_text_2, send EXACTLY: {template_text!r}",
                "  Do NOT add any other text.",
            ]
        elif action.template_key == "ask_topic_clarification" and not state.get("active_topics"):
            lines += [
                "  For reply_text_2: customer hasn't stated a specific need.",
                "  Greet warmly + invite them to share what they're looking for.",
                "  Use time-based greeting from business context above.",
                f"  {gender_note}",
            ]
        else:
            lines += [
                f"  For reply_text_2, ask: {template_text!r}",
                f"  {gender_note}",
            ]
        lines += [
            "  ⛔ reply_text must be EXACTLY the pitch text — no changes.",
        ]

    # ── Fixed messages (non-first) ────────────────────────────────────────────
    elif action.is_fixed:
        if action.template_key in ("stage3_question", "stage3_question_female", "stage3_question_male"):
            stage3_text = QUESTION_TEMPLATES.get(action.template_key, STAGE3_QUESTION)
            flat_note = (
                "  ⛔ entrance_style=flat — there is NO catalog for flat doors. Do NOT invent or send any URL."
                if state.get("entrance_style") == "flat" or state.get("interior_style") == "flat"
                else ""
            )
            lines += [
                f'INSTRUCTION: Send EXACTLY this text in reply_text: {stage3_text!r}',
                "  ⛔ Do NOT add ANY text before or after it.",
                "  ⛔ Do NOT include any URLs, catalog links, or website addresses — not even michaeldoors.co.il.",
                "  ⛔ Catalog sending is handled by a SEPARATE action — never send a catalog here.",
                *(([flat_note]) if flat_note else []),
                "  reply_text_2: null",
            ]
        elif action.template_key in ("contact_opener", "contact_opener_showroom"):
            opener = QUESTION_TEMPLATES.get(action.template_key, CONTACT_OPENER)
            lines += [
                f'INSTRUCTION: Send EXACTLY this text in reply_text: {opener!r}',
                "  ⛔ Do NOT append any question to the opener — send it ALONE.",
                "  ⛔ Wait for customer reply before asking phone/name/city.",
                "  reply_text_2: null",
            ]
        elif action.template_key in ("entrance_catalog", "interior_catalog"):
            lines += [
                f'INSTRUCTION: Send EXACTLY this text in reply_text: {template_text!r}',
                "  ⛔ reply_text must be EXACTLY the catalog text above — do not change a single character.",
            ]
            nxt, nxt_topic = _peek_catalog_next_action(action, state)
            if nxt:
                nxt_template = QUESTION_TEMPLATES.get(nxt.template_key, "")
                dst_label = _TOPIC_LABELS_HE.get(nxt_topic or "", "")
                transition = f"נעבור ל{dst_label} — " if dst_label else ""
                lines += [
                    f"  reply_text_2: Short follow-up that transitions to the next topic.",
                    f"  Send EXACTLY: '{transition}{nxt_template}'",
                    f"  (Adapt gender forms per {gender_note} if needed, but keep the question exact.)",
                ]
            else:
                lines += ["  reply_text_2: null"]
            lines += [""]
        elif action.template_key == "_farewell_dynamic":
            farewell_text = _get_farewell_text(state)
            lines += [
                f'INSTRUCTION: Send EXACTLY this text in reply_text: {farewell_text!r}',
                "  ⛔ Do NOT change a single word. Do NOT add names, times, blessings, or any other text.",
                "  ⛔ Do NOT write a custom farewell — use only the exact text above.",
                "  Set handoff_to_human: true",
                "  reply_text_2: null",
            ]
        elif action.template_key in (
            "ask_callback_time_neutral", "ask_callback_time_female", "ask_callback_time_male",
            "ask_callback_time_showroom_neutral", "ask_callback_time_showroom_female", "ask_callback_time_showroom_male",
        ):
            lines += [
                f'INSTRUCTION: Send EXACTLY this text in reply_text: {template_text!r}',
                "  ⛔ Do NOT rephrase, add names, or change a single word.",
                "  reply_text_2: null",
            ]
        else:
            lines += [
                f'INSTRUCTION: Send EXACTLY: {template_text!r}',
                "  reply_text_2: null",
            ]

    # ── Near-miss phone — targeted correction ────────────────────────────────
    elif action.field_to_ask == "phone" and state.get("_near_miss_phone"):
        near_miss = state["_near_miss_phone"]
        digit_count = len(re.sub(r'\D', '', near_miss))
        lines += [
            f"INSTRUCTION: The customer sent '{near_miss}' which has only {digit_count} digits — it looks like a phone number with a missing digit.",
            f"  Ask them to re-send their full phone number, mentioning '{near_miss}' as the number you received.",
            f"  Example (adapt gender per {gender_note}): \"נראה שחסרה ספרה במספר {near_miss}, תוכל/תוכלי/תוכלו לשלוח שוב?\"",
            "  Keep it short (1–2 lines). Do NOT ask for any other field.",
            f"  {gender_note}",
        ]

    # ── Dynamic questions ─────────────────────────────────────────────────────
    elif action.template_key == "_summary_dynamic":
        name = (state.get("full_name") or "לקוח/ה").split()[0]
        topic_label = _build_service_label_he(state)  # built from state fields — never exposes internal keys
        lines += [
            "INSTRUCTION: Stage 5 — Send a summary and ask for confirmation.",
            f"  Open with a SHORT warm greeting using the customer's first name, e.g.: 'מדהים, {name} 😊'",
            "  ⛔ Do NOT write 'הכל נכון' in the opening — it must appear ONLY at the very end.",
            "  After the greeting, list the details exactly as follows (one field per line, no extras):",
            f"    נושא הפנייה: {topic_label}",
            f"    שם: {state.get('full_name')}",
            f"    עיר: {state.get('city')}",
            f"    טלפון: {state.get('phone')}",
            '  Close the message with EXACTLY: "הכל נכון?"',
            "  ⛔ 'הכל נכון?' appears EXACTLY ONCE — as the last line only. Never at the start.",
            "  handoff_to_human: false (wait for customer confirmation)",
            "  reply_text_2: null",
        ]
    else:
        lines += [
            f"INSTRUCTION: Ask this question in reply_text (adapt wording naturally):",
            f"  Template: {template_text!r}",
            "",
            "Rules:",
            "  - Ask ONLY this one question. Zero other questions.",
            "  - 1–3 lines max. WhatsApp style.",
            "  - A brief warm acknowledgment of the customer's last message is allowed",
            "    (max 1 line), then ask the question directly.",
            f"  - {gender_note}",
        ]

    # ── Fields already collected ──────────────────────────────────────────────
    collected = []
    if state.get("phone"):
        collected.append(f"phone={state['phone']}")
    if state.get("full_name"):
        collected.append(f"name={state['full_name']}")
    if state.get("city"):
        collected.append(f"city={state['city']}")
    if collected:
        lines.append(f"\n⛔ Already collected (never re-ask): {', '.join(collected)}")

    return "\n".join(lines)


# ── Business context ──────────────────────────────────────────────────────────
_BUSINESS = {
    "name":    "Michael Doors",
    "phone":   "054-2787578",
    "products": [
        "Entrance doors (smooth & designed): Nefachim, Panoramic, Greek, Mercury, Eden, Eden Brass series",
        "Interior doors: smooth, grooves, squares, arc, cross styles",
        "Hardware & handles",
        "Mamad (security room) doors",
        "Institutional & warehouse doors",
    ],
    "hours": {"start": 9, "end": 18, "tz": "Asia/Jerusalem", "days": "א'–ה'", "fri_end": 13, "closed": "שבת וחגים"},
}


def is_working_hours() -> bool:
    from datetime import datetime
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(_BUSINESS["hours"]["tz"]))
    hour, weekday = now.hour, now.weekday()
    if weekday == 5:
        return False
    if weekday == 4:
        return _BUSINESS["hours"]["start"] <= hour < _BUSINESS["hours"]["fri_end"]
    return _BUSINESS["hours"]["start"] <= hour < _BUSINESS["hours"]["end"]


def _context_block() -> str:
    from datetime import datetime
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(_BUSINESS["hours"]["tz"]))
    weekday = now.weekday()
    if weekday == 5:
        status = "outside working hours (Saturday — closed)"
    elif weekday == 4:
        status = (
            f"within working hours (Friday {_BUSINESS['hours']['start']}:00–{_BUSINESS['hours']['fri_end']}:00)"
            if is_working_hours()
            else f"outside working hours (Friday closes at {_BUSINESS['hours']['fri_end']}:00)"
        )
    else:
        status = (
            f"within working hours ({_BUSINESS['hours']['start']}:00–{_BUSINESS['hours']['end']}:00)"
            if is_working_hours()
            else "outside working hours — let the customer know and offer to schedule a callback"
        )
    return "\n".join([
        f"Business: {_BUSINESS['name']}",
        f"Phone: {_BUSINESS['phone']}",
        f"Products: {', '.join(_BUSINESS['products'])}",
        "Hours: Sun–Thu 09:00–18:00 | Fri 09:00–13:00 | Sat closed",
        f"Current time status: {status}",
    ])


def _israel_greeting() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    hour = datetime.now(ZoneInfo("Asia/Jerusalem")).hour
    if 6 <= hour < 12:
        return "בוקר טוב"
    elif 12 <= hour < 17:
        return "צהריים טובים"
    elif 17 <= hour < 21:
        return "ערב טוב"
    else:
        return "לילה טוב"


def _next_opening_time() -> str:
    from datetime import datetime, timedelta
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(_BUSINESS["hours"]["tz"]))
    wd = now.weekday()
    hour = now.hour
    h_start = _BUSINESS["hours"]["start"]
    h_end   = _BUSINESS["hours"]["end"]
    h_fri   = _BUSINESS["hours"]["fri_end"]
    if wd < 4 and hour < h_end:
        return f"היום עד {h_end}:00"
    if wd == 4 and hour < h_fri:
        return f"היום עד {h_fri}:00"
    if wd == 4 and hour >= h_fri:
        return f"ביום ראשון משעה {h_start}:00"
    if wd == 5:
        return f"ביום ראשון משעה {h_start}:00"
    if wd == 6 and hour < h_end:
        return f"היום עד {h_end}:00"
    tomorrow_he = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
    next_day = tomorrow_he[(wd + 1) % 7]
    return f"ביום {next_day} משעה {h_start}:00"


# ── FAQ helpers ───────────────────────────────────────────────────────────────
def _find_faqs(user_msg: str) -> list[dict]:
    msg = user_msg.lower()
    matched = [
        e for e in _faq_bank
        # Exclude language_* entries — non-Hebrew is handled at the Python gate
        if not e.get("category", "").startswith("language_")
        and any(kw.lower() in msg for kw in e.get("keywords", []))
    ]
    return matched[:3]


def _faq_block(faqs: list[dict]) -> str | None:
    if not faqs:
        return None
    lines = [f"[{f['category']}] {f['answer']}" for f in faqs]
    return "## מידע רלוונטי מבסיס הידע (לשימוש כהפניה בלבד — אל תעתיק את הניסוח)\n" + "\n".join(lines)


def _build_system(
    user_msg: str,
    sender: str,
    state: dict,
    history: list[dict],
    action: NextAction,
    is_first_message: bool,
) -> str:
    if not _SYSTEM_PROMPT:
        logger.error("System prompt is empty — Claude will have no instructions")
    greeting = _israel_greeting()
    parts = [
        _SYSTEM_PROMPT,
        f"## Business context\n{_context_block()}",
        (
            f"## Current time context\nCurrent greeting for this time of day: «{greeting}»\n"
            "Use this greeting in reply_text_2 on the FIRST reply only.\n"
            "On all subsequent replies: do NOT include any time-based greeting."
        ),
        _state_summary_block(state),
    ]

    _is_bypass = sender and sender in _cfg.HOURS_BYPASS_PHONES
    if not is_working_hours() and not _is_bypass:
        next_open = _next_opening_time()
        parts.append(
            "## OUT-OF-HOURS — MANDATORY BEHAVIOUR\n"
            f"The business is currently CLOSED. Next opening: {next_open}.\n"
            "Acknowledge this in your reply. Include:\n"
            "1. We are not available right now but received the message.\n"
            f"2. We will call back {next_open}.\n"
            "3. Customer can call directly: 054-2787578.\n"
            "Still collect name/phone/city — sales manager reviews leads in the morning."
        )

    parts.append(
        "## ABSOLUTE RULE — PRICE/DELIVERY DISCLOSURE FORBIDDEN\n"
        "NEVER state, estimate, hint at, or compare any price, price range, cost, or delivery time. "
        "This rule overrides every other instruction. "
        "If asked about price: "
        "'המחיר מותאם אישית לפי סוג ועיצוב — אשמח שתשאירו פרטים ונחזור עם הצעה מסודרת 😊'"
    )

    # Suppress FAQ for fixed-message actions — Claude must send EXACTLY the
    # template text and must not append URLs or extra info from the knowledge base.
    if not action.is_fixed:
        faqs = _find_faqs(user_msg)
        block = _faq_block(faqs)
        if block:
            parts.append(block)
            logger.info("FAQ match: %s", ", ".join(f["id"] for f in faqs))

    # ── DECIDED ACTION block — injected LAST (highest recency in context) ─────
    parts.append(_build_action_block(action, state, is_first_message))

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# AI CALLING
# ══════════════════════════════════════════════════════════════════════════════

_claude: anthropic.AsyncAnthropic | None = None
_openrouter = None

_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
_CLAUDE_MODEL     = "claude-sonnet-4-6"


def _use_openrouter() -> bool:
    return bool(_cfg.OPENROUTER_API_KEY)


def _get_claude(api_key: str) -> anthropic.AsyncAnthropic:
    global _claude
    if _claude is None:
        _claude = anthropic.AsyncAnthropic(api_key=api_key)
    return _claude


def _get_openrouter():
    global _openrouter
    if _openrouter is None:
        from openai import AsyncOpenAI
        _openrouter = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_cfg.OPENROUTER_API_KEY,
        )
    return _openrouter


async def _call_ai(system: str, messages: list, max_tokens: int, api_key: str, timeout: float = 50.0) -> str:
    global DIAG_STATE
    if _use_openrouter():
        try:
            client = _get_openrouter()
            response = await client.chat.completions.create(
                model=_OPENROUTER_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}] + messages,
                timeout=timeout,
            )
            content = response.choices[0].message.content or ""
            if content:
                DIAG_STATE["last_ai_provider"] = f"openrouter/{_OPENROUTER_MODEL}"
                return content
            raise ValueError("OpenRouter returned empty content")
        except Exception as or_exc:
            logger.warning("[OPENROUTER:FAIL] %s — falling back to Claude", or_exc)
            DIAG_STATE["openrouter_failures"] = DIAG_STATE.get("openrouter_failures", 0) + 1
            DIAG_STATE["last_openrouter_error"] = str(or_exc)[:200]

    client = _get_claude(api_key)
    response = await client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        timeout=timeout,
    )
    DIAG_STATE["last_ai_provider"] = f"claude/{_CLAUDE_MODEL}"
    return response.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ══════════════════════════════════════════════════════════════════════════════

_PARSE_ERROR_REPLY = _ERR["parse_error"]
_API_ERROR_REPLY   = _ERR["api_error"]
ERROR_REPLIES: frozenset[str] = frozenset([_PARSE_ERROR_REPLY, _API_ERROR_REPLY])

_MAX_HISTORY = 40

_PRICE_RE = re.compile(
    r'(?:כ[-–]?|מ[-–]?|ב[-–]?|עד\s)?'
    r'\d[\d,\.]*\s*(?:₪|ש["\']?ח\b|שקל\b)'
    r'|(?:₪)\s*\d[\d,\.]*',
    re.UNICODE,
)


def _scrub_prices(text: str, sender: str) -> str:
    if not _PRICE_RE.search(text):
        return text
    scrubbed = _PRICE_RE.sub("מחיר מותאם אישית", text)
    logger.warning("[PRICE:BLOCKED] scrubbed | sender=%s | original=%s", sender, text[:100])
    return scrubbed


def _extract_json(text: str) -> str:
    marker = '"reply_text"'
    last_pos = text.rfind(marker)
    if last_pos > 0:
        brace_pos = text.rfind("{", 0, last_pos)
        if brace_pos >= 0:
            candidate = text[brace_pos:]
            depth = 0
            for i, ch in enumerate(candidate):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return candidate[:i + 1]
    brace_pos = text.find("{")
    if brace_pos >= 0:
        return text[brace_pos:]
    return text


def _parse_response(raw: str, sender: str) -> dict:
    try:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = _extract_json(cleaned)
        parsed  = json.loads(cleaned)

        reply_text = str(parsed.get("reply_text", ""))
        if not reply_text.strip():
            reply_text = _PARSE_ERROR_REPLY
        reply_text = _scrub_prices(reply_text, sender)

        reply_text_2_raw = parsed.get("reply_text_2")
        reply_text_2 = str(reply_text_2_raw).strip() if reply_text_2_raw else None
        if reply_text_2:
            reply_text_2 = _scrub_prices(reply_text_2, sender)

        return {
            "reply_text":     reply_text,
            "reply_text_2":   reply_text_2,
            "handoff_to_human": bool(parsed.get("handoff_to_human", False)),
            "summary":        str(parsed.get("summary", "")),

            # Claude's extracted fields (new v2 schema)
            "extracted_full_name":             parsed.get("extracted_full_name"),
            "extracted_phone":                 parsed.get("extracted_phone"),
            "extracted_city":                  parsed.get("extracted_city"),
            "extracted_preferred_contact_hours": parsed.get("extracted_preferred_contact_hours"),
            "extracted_entrance_scope":        parsed.get("extracted_entrance_scope"),
            "extracted_entrance_style":        parsed.get("extracted_entrance_style"),
            "extracted_entrance_model":        parsed.get("extracted_entrance_model"),
            "extracted_interior_project_type": parsed.get("extracted_interior_project_type"),
            "extracted_interior_quantity":     parsed.get("extracted_interior_quantity"),
            "extracted_interior_style":        parsed.get("extracted_interior_style"),
            "extracted_interior_model":        parsed.get("extracted_interior_model"),
            "extracted_mamad_type":            parsed.get("extracted_mamad_type"),
            "extracted_customer_gender_locked": parsed.get("extracted_customer_gender_locked"),
            "extracted_service_type":          parsed.get("extracted_service_type"),
            "extracted_showroom_requested":    parsed.get("extracted_showroom_requested"),
            "detected_new_topics":             parsed.get("detected_new_topics") or [],
        }

    except Exception:
        plain = raw.strip()
        if plain and plain not in (_PARSE_ERROR_REPLY, _API_ERROR_REPLY):
            logger.warning("Non-JSON response — using plain text | sender=%s | raw: %s", sender, raw[:120])
            plain = _scrub_prices(plain, sender)
            return _parse_fallback(plain)
        logger.warning("Non-JSON empty response | sender=%s | raw: %s", sender, raw[:120])
        return _parse_fallback(_PARSE_ERROR_REPLY)


def _parse_fallback(reply_text: str) -> dict:
    return {
        "reply_text": reply_text, "reply_text_2": None,
        "handoff_to_human": False, "summary": "Parse fallback",
        "extracted_full_name": None, "extracted_phone": None,
        "extracted_city": None, "extracted_preferred_contact_hours": None,
        "extracted_entrance_scope": None, "extracted_entrance_style": None,
        "extracted_entrance_model": None, "extracted_interior_project_type": None,
        "extracted_interior_quantity": None, "extracted_interior_style": None,
        "extracted_interior_model": None, "extracted_mamad_type": None,
        "extracted_customer_gender_locked": None,
        "extracted_service_type": None, "extracted_showroom_requested": None,
        "detected_new_topics": [],
    }


def _extract_claude_fields(structured: dict) -> dict:
    """Map Claude's extracted_* fields to state field names."""
    mapping = {
        "extracted_full_name":             "full_name",
        "extracted_phone":                 "phone",
        "extracted_city":                  "city",
        "extracted_preferred_contact_hours": "preferred_contact_hours",
        "extracted_entrance_scope":        "entrance_scope",
        "extracted_entrance_style":        "entrance_style",
        "extracted_entrance_model":        "entrance_model",
        "extracted_interior_project_type": "interior_project_type",
        "extracted_interior_quantity":     "interior_quantity",
        "extracted_interior_style":        "interior_style",
        "extracted_interior_model":        "interior_model",
        "extracted_mamad_type":            "mamad_type",
        "extracted_customer_gender_locked": "customer_gender_locked",
        "extracted_service_type":          "service_type",
    }
    fields: dict = {}
    for ex_key, state_key in mapping.items():
        val = structured.get(ex_key)
        if val is not None:
            fields[state_key] = val

    new_topics = structured.get("detected_new_topics") or []
    if new_topics:
        fields["_new_topics"] = new_topics

    if structured.get("extracted_showroom_requested"):
        fields["showroom_requested"] = True

    return fields


def _sanitize_input(text: str, sender: str) -> str:
    if len(text) > _MAX_INPUT_CHARS:
        logger.warning("[INPUT:TRUNCATE] %d→%d | sender=%s", len(text), _MAX_INPUT_CHARS, sender)
        text = text[:_MAX_INPUT_CHARS]
    return text


def _validate_history(sender: str) -> None:
    history = _conversations.get(sender, [])
    valid = [
        m for m in history
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]
    if len(valid) != len(history):
        logger.warning("[HIST:FIX] Removed %d malformed entries | sender=%s",
                       len(history) - len(valid), sender)
        _conversations[sender] = valid


# ── Last-seen timestamps ──────────────────────────────────────────────────────
_last_seen: dict[str, float] = {}
try:
    _last_seen = json.loads(_LAST_SEEN_PATH.read_text(encoding="utf-8"))
except Exception:
    pass


def _save_last_seen() -> None:
    try:
        _LAST_SEEN_PATH.write_text(json.dumps(_last_seen), encoding="utf-8")
    except Exception:
        pass


_force_fresh: set[str] = set()


async def _supabase_save_conv(sender: str) -> None:
    try:
        from ..providers.supabase_store import save_conversation
        await save_conversation(sender, _conversations.get(sender, []))
    except Exception as e:
        logger.warning("[SUPABASE] save_conversation failed: %s", e)


def clear_conversation(sender: str) -> None:
    """Force a fresh session on the next get_reply call for this sender."""
    _force_fresh.add(sender)


def _empty_return(reply_text: str, summary: str, state: dict | None = None) -> dict:
    """Build a minimal return dict compatible with main.py's expected keys."""
    s = state or {}
    scope = s.get("entrance_scope")
    return {
        "reply_text":              reply_text,
        "reply_text_2":            None,
        "handoff_to_human":        False,
        "summary":                 summary,
        "preferred_contact_hours": s.get("preferred_contact_hours"),
        "needs_frame_removal":     (True  if scope == "with_frame" else
                                    False if scope == "door_only"  else None),
        "needs_installation":      None,
        "full_name":               s.get("full_name"),
        "phone":                   s.get("phone"),
        "service_type":            s.get("service_type"),
        "city":                    s.get("city"),
        # Legacy names kept for _record_lead compat
        "doors_count":             s.get("interior_quantity"),
        "design_preference":       (s.get("entrance_style") or s.get("interior_style")),
        "project_status":          s.get("interior_project_type"),
        "referral_source":         s.get("referral_source"),
        "is_returning_customer":   s.get("is_returning_customer"),
        "active_topics":           s.get("active_topics", []),
        "current_active_topic":    s.get("current_active_topic"),
        # v2 door detail fields (for richer Sheets payload)
        "entrance_scope":          scope,
        "entrance_style":          s.get("entrance_style"),
        "entrance_model":          s.get("entrance_model"),
        "interior_project_type":   s.get("interior_project_type"),
        "interior_quantity":       s.get("interior_quantity"),
        "interior_style":          s.get("interior_style"),
        "interior_model":          s.get("interior_model"),
        "mamad_type":              s.get("mamad_type"),
    }


def _structured_to_return(structured: dict, state: dict) -> dict:
    """Convert parsed Claude response + state into the return dict for main.py."""
    s = state or {}
    scope = s.get("entrance_scope")
    return {
        "reply_text":              structured["reply_text"],
        "reply_text_2":            structured.get("reply_text_2"),
        "handoff_to_human":        structured.get("handoff_to_human", False),
        "summary":                 structured.get("summary", ""),
        "preferred_contact_hours": s.get("preferred_contact_hours"),
        "needs_frame_removal":     (True  if scope == "with_frame" else
                                    False if scope == "door_only"  else None),
        "needs_installation":      None,
        "full_name":               s.get("full_name"),
        "phone":                   s.get("phone"),
        "service_type":            s.get("service_type"),
        "city":                    s.get("city"),
        # Legacy names kept for _record_lead compat
        "doors_count":             s.get("interior_quantity"),
        "design_preference":       (s.get("entrance_style") or s.get("interior_style")),
        "project_status":          s.get("interior_project_type"),
        "referral_source":         s.get("referral_source"),
        "is_returning_customer":   s.get("is_returning_customer"),
        "active_topics":           s.get("active_topics", []),
        "current_active_topic":    s.get("current_active_topic"),
        # v2 door detail fields (for richer Sheets payload)
        "entrance_scope":          scope,
        "entrance_style":          s.get("entrance_style"),
        "entrance_model":          s.get("entrance_model"),
        "interior_project_type":   s.get("interior_project_type"),
        "interior_quantity":       s.get("interior_quantity"),
        "interior_style":          s.get("interior_style"),
        "interior_model":          s.get("interior_model"),
        "mamad_type":              s.get("mamad_type"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

async def get_reply(
    sender: str,
    user_message: str,
    anthropic_api_key: str,
    mock_claude: bool = False,
) -> dict:
    import time as _time

    user_message = _sanitize_input(user_message, sender)

    # ── Session management ─────────────────────────────────────────────────────
    now = _time.time()
    if sender in _force_fresh:
        _force_fresh.discard(sender)
        _conversations.pop(sender, None)
        _conv_state.pop(sender, None)
        _last_seen.pop(sender, None)
        logger.info("[SESSION:FORCED] Fresh start | sender=%s", sender)
    elif not _cfg.TEST_MODE:
        last = _last_seen.get(sender, 0.0)
        if last > 0 and (now - last) > _SESSION_GAP:
            gap_h = (now - last) / 3600
            logger.info("[SESSION:RESET] %.1fh gap — fresh start | sender=%s", gap_h, sender)
            _conversations.pop(sender, None)
            _conv_state.pop(sender, None)
    _last_seen[sender] = now
    _save_last_seen()

    # ── History management ─────────────────────────────────────────────────────
    if sender not in _conversations:
        _conversations[sender] = []
    _conversations[sender].append({"role": "user", "content": user_message})
    if len(_conversations[sender]) > _MAX_HISTORY:
        _conversations[sender] = _conversations[sender][-_MAX_HISTORY:]
        logger.info("[HIST:TRIM] Trimmed to %d turns | sender=%s", _MAX_HISTORY, sender)
    _validate_history(sender)

    # ── State initialization / migration ──────────────────────────────────────
    if sender not in _conv_state or not _is_v2_state(_conv_state[sender]):
        _conv_state[sender] = _empty_conv_state()
        logger.info("[STATE:INIT] Fresh v2 state | sender=%s", sender)

    state   = _conv_state[sender]
    history = _conversations[sender]
    is_first_message = len(history) == 1  # only the user message we just appended

    # ── Mock mode ──────────────────────────────────────────────────────────────
    if mock_claude:
        turn = len(history)
        mock_reply = f"🤖 [מוק סיבוב {turn}] AI היה עונה כאן על: ״{user_message[:40]}״"
        history.append({"role": "assistant", "content": mock_reply})
        _save_conversations()
        return _empty_return(mock_reply, f"Mock mode turn {turn}", state)

    # ── Hebrew-only gate ──────────────────────────────────────────────────────
    # If the message contains ONLY foreign letters (no Hebrew, no digits) → return fixed Hebrew reply.
    # This covers: pure English, pure Russian, pure Arabic, etc.
    # Mixed messages (Hebrew + another language), digit-only, emoji-only, and punctuation all pass through.
    # The same reply is returned every time they write non-Hebrew — no language switch.
    if _needs_hebrew_fallback(user_message):
        logger.info("[LANG:NON-HEBREW] Returning Hebrew fallback | sender=%s | msg=%s",
                    sender, user_message[:60])
        history.append({"role": "assistant", "content": _HEBREW_ONLY_REPLY})
        _save_conversations()
        asyncio.create_task(_supabase_save_conv(sender))
        return _empty_return(_HEBREW_ONLY_REPLY, "Non-Hebrew input — Hebrew fallback", state)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    # Step 1: Extract fields from customer message (regex)
    extracted = _extract_fields_from_message(user_message, state)

    # Step 2: Detect new topics from message
    new_topics = _detect_topics_from_message(user_message)
    if new_topics:
        extracted["_new_topics"] = new_topics
        logger.info("[TOPICS:DETECT] sender=%s | %s", sender, new_topics)

    # Step 3: Merge extracted fields into state
    state = _merge_state(state, extracted)
    _conv_state[sender] = state

    # Clear near-miss marker once a valid phone has been collected by regex
    if state.get("phone") and state.get("_near_miss_phone"):
        state["_near_miss_phone"] = None
        logger.info("[NEAR_MISS:CLEAR] Valid phone collected via regex | sender=%s", sender)

    if extracted:
        logger.info("[EXTRACT:REGEX] sender=%s | %s", sender, {k: v for k, v in extracted.items() if k != "_new_topics"})

    # Step 4: Apply buffered style to current topic
    _apply_style_to_topic(state)

    # Step 5: Advance stage flags (reads history)
    _advance_stage(state, history)

    # Step 6: Decide next action (pure state function)
    action = _decide_next_action(state)
    logger.info("[ACTION] sender=%s | stage=%d | field=%s | template=%s | context=%s",
                sender, action.stage, action.field_to_ask, action.template_key, action.context)

    # Save state before AI call
    _save_conv_state()

    # ── AI call ────────────────────────────────────────────────────────────────
    provider = "openrouter" if _use_openrouter() else "claude"
    system = _build_system(user_message, sender, state, history, action, is_first_message)

    try:
        _t0 = _time.monotonic()
        logger.info("[AI:REQ] provider=%s | sender=%s | turns=%d | action=%s",
                    provider, sender, len(history), action.field_to_ask)
        raw_text = None
        for attempt in range(3):
            try:
                raw_text = await _call_ai(
                    system=system,
                    messages=history,
                    max_tokens=900,
                    api_key=anthropic_api_key,
                    timeout=50.0,
                )
                break
            except (anthropic.RateLimitError, anthropic.APITimeoutError) as retry_exc:
                if attempt == 2:
                    raise
                wait = 5 * (2 ** attempt)
                logger.warning("[AI:RETRY] attempt=%d | %s — waiting %ds", attempt + 1, retry_exc, wait)
                await asyncio.sleep(wait)
        if not raw_text:
            raise ValueError("AI returned empty response")
        logger.info("[AI:OK] provider=%s | sender=%s | latency=%.1fs",
                    provider, sender, _time.monotonic() - _t0)
    except Exception as exc:
        logger.error("[AI:ERR] sender=%s | %s", sender, exc)
        fallback = _API_ERROR_REPLY
        history.append({"role": "assistant", "content": fallback})
        _save_conversations()
        return _empty_return(fallback, "AI error", state)

    # ── Parse response ─────────────────────────────────────────────────────────
    structured = _parse_response(raw_text, sender)

    # ── Merge Claude's extracted fields into state ─────────────────────────────
    claude_fields = _extract_claude_fields(structured)
    state = _merge_state(state, claude_fields)
    _conv_state[sender] = state

    # Clear near-miss marker once a valid phone has been collected by Claude
    if state.get("phone") and state.get("_near_miss_phone"):
        state["_near_miss_phone"] = None
        logger.info("[NEAR_MISS:CLEAR] Valid phone collected via Claude | sender=%s", sender)

    # ── Stage 7 safety: hard-override farewell text ───────────────────────────
    # Claude sometimes adds names, blessings, or times around the farewell.
    # Overriding here guarantees the customer always receives the exact template.
    if action.field_to_ask == "farewell":
        structured["reply_text"] = _get_farewell_text(state)
        structured["reply_text_2"] = None
        structured["handoff_to_human"] = True
        logger.info("[FAREWELL:OVERRIDE] sender=%s | text=%s", sender, structured["reply_text"])

    # ── Post-call: store reply in history, then re-advance stage flags ─────────
    history_content = structured["reply_text"]
    if structured.get("reply_text_2"):
        history_content += "\n\n" + structured["reply_text_2"]
    history.append({"role": "assistant", "content": history_content})

    _advance_stage(state, history)  # catch any new flags set by this reply
    _conv_state[sender] = state

    # ── First-message safety: ensure reply_text == PITCH ──────────────────────
    if is_first_message:
        structured["reply_text"] = PITCH

    # ── Follow-up turns: strip PITCH if Claude accidentally included it ────────
    if not is_first_message:
        _GREETING_PAT = re.compile(
            r"(?:היי,?\s*)?תודה שפניתם (?:לדלתות מיכאל|למיכאל דלתות)[^\n]*\n?",
            re.IGNORECASE,
        )
        _PITCH_PAT = re.compile(
            r"(?:אנחנו|אנו) מציעים דלתות כניסה ופנים[^\n]*\n?",
            re.IGNORECASE,
        )
        def _strip_pitch(text: str) -> str:
            text = _GREETING_PAT.sub("", text).strip()
            text = _PITCH_PAT.sub("", text).strip()
            return text

        stripped = _strip_pitch(structured["reply_text"])
        if stripped:
            structured["reply_text"] = stripped
        # Preserve reply_text_2 for catalog actions — they use it to send the
        # immediate next-topic question in the same turn.
        if action.template_key not in ("entrance_catalog", "interior_catalog"):
            structured["reply_text_2"] = None

    # ── Log ───────────────────────────────────────────────────────────────────
    if structured["reply_text"] in ERROR_REPLIES:
        logger.warning("[FALLBACK] Parse fallback | sender=%s | raw=%s", sender, raw_text[:80])
    else:
        logger.info("[REPLY:OK] sender=%s | text=%s", sender, structured["reply_text"][:60])

    # ── Persist ───────────────────────────────────────────────────────────────
    _save_conversations()
    asyncio.create_task(_supabase_save_conv(sender))
    _save_conv_state()

    return _structured_to_return(structured, state)


# ── Follow-up message (15-min silence reminder) ───────────────────────────────
async def get_followup_message(sender: str, anthropic_api_key: str) -> str:
    history = _conversations.get(sender, [])
    _FALLBACK = "היי, עדיין ממתינה לתגובה מכם 😊 אם יש שאלה נוספת, אנחנו כאן לעזור!"
    if len(history) < 2:
        _conversations.setdefault(sender, []).append({"role": "assistant", "content": _FALLBACK})
        _save_conversations()
        return _FALLBACK
    system = (
        "אתה נציג מכירות של דלתות מיכאל. "
        "הלקוח לא ענה כבר 15 דקות. כתוב הודעת תזכורת קצרה בשורה אחת עד שתיים: "
        "\"היי, עדיין ממתינה לתגובה מכם 😊 אם יש עוד שאלות לגבי [נושא ספציפי מהשיחה], אנחנו כאן!\". "
        "החלף [נושא ספציפי] בנושא מהשיחה. "
        "שפה ישירה ואנושית. בעברית בלבד. ללא JSON."
    )
    try:
        msg = await _call_ai(system=system, messages=history[-6:], max_tokens=120,
                             api_key=anthropic_api_key, timeout=15.0)
        msg = msg.strip()
        _conversations[sender].append({"role": "assistant", "content": msg})
        _save_conversations()
        return msg
    except Exception as exc:
        logger.error("get_followup_message error | sender=%s | %s", sender, exc)
        _conversations.setdefault(sender, []).append({"role": "assistant", "content": _FALLBACK})
        _save_conversations()
        return _FALLBACK


# ── Closing intent detection ─────────────────────────────────────────────────
def is_closing_intent(text: str, conv_turns: int) -> bool:
    """
    Return True if the customer's message looks like a goodbye/closing intent.
    Only fires after at least 2 turns so first-message greetings aren't treated as closings.
    """
    if conv_turns < 2:
        return False
    stripped = text.strip()
    # Short standalone farewell (≤30 chars covers "תודה", "ביי", "תודה רבה", "להתראות")
    if len(stripped) <= 30 and re.search(
        r'תודה|ביי|להתראות|יום טוב|לילה טוב|שבוע טוב|חג שמח|שנה טובה|עד הפעם|סיימנו|הבנתי תודה',
        stripped, re.IGNORECASE
    ):
        return True
    # Longer text that is explicitly a goodbye
    if re.search(
        r'^(?:אוקי\s+)?תודה(?:\s+רבה)?[.!]?\s*(?:ביי|להתראות|יום טוב)?$',
        stripped, re.IGNORECASE
    ):
        return True
    return False


# ── Closing message (farewell AI reply) ───────────────────────────────────────
async def get_closing_message(sender: str, anthropic_api_key: str) -> str:
    """Generate a warm farewell message when the customer closes the conversation."""
    history = _conversations.get(sender, [])
    _FALLBACK = "תודה שפניתם לדלתות מיכאל 😊 אם תרצו לחזור — אנחנו כאן! יום נפלא! 💙"
    system = (
        "אתה נציג מכירות ידידותי של דלתות מיכאל. "
        "הלקוח מסיים את השיחה. כתוב הודעת פרידה קצרה (1–2 שורות), חמה ואנושית. "
        "אם נמסרו פרטי קשר, ציין שנחזור בהקדם. "
        "בעברית בלבד. ללא JSON."
    )
    try:
        msg = await _call_ai(
            system=system,
            messages=(history[-4:] if history else [{"role": "user", "content": "להתראות"}]),
            max_tokens=120,
            api_key=anthropic_api_key,
            timeout=15.0,
        )
        return msg.strip() or _FALLBACK
    except Exception as exc:
        logger.error("get_closing_message error | sender=%s | %s", sender, exc)
        return _FALLBACK


# ── Conversation summary (called at conversation close) ───────────────────────
async def generate_conversation_summary(sender: str, anthropic_api_key: str) -> str:
    """Generate a concise summary of the completed conversation for the lead record."""
    history = _conversations.get(sender, [])
    _FALLBACK = "שיחה ללא סיכום"
    if not history:
        return _FALLBACK
    system = (
        "סכם את שיחת המכירה הבאה בנקודות קצרות (עברית):\n"
        "• מה הלקוח חיפש (סוג דלת, כמות, עיצוב)\n"
        "• פרטי קשר שנמסרו (שם, עיר, טלפון, זמן חזרה)\n"
        "• שלב השיחה בו הסתיימה\n"
        "3–6 שורות. ללא JSON."
    )
    try:
        msg = await _call_ai(
            system=system,
            messages=history[-20:],
            max_tokens=350,
            api_key=anthropic_api_key,
            timeout=20.0,
        )
        return msg.strip() or _FALLBACK
    except Exception as exc:
        logger.error("generate_conversation_summary error | sender=%s | %s", sender, exc)
        return _FALLBACK


# ── Public API for main.py ─────────────────────────────────────────────────────
def get_conversations() -> dict:
    return _conversations


def get_conv_state() -> dict:
    return _conv_state
