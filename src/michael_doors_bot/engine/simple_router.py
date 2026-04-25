"""
Scenario classifier + Claude service.
Scenario classifier runs only on the first message of a conversation.
All subsequent messages go directly to Claude.
"""
import asyncio
import json
import logging
import re
from pathlib import Path

import anthropic
from .messages import SCENARIO_RESPONSES as _MSG, ERROR_MSG as _ERR

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent.parent.parent.parent
_PROMPT_PATH  = _ROOT / "src" / "prompts" / "systemPrompt.txt"
_FAQ_PATH     = _ROOT / "src" / "data" / "faqBank.json"

# DATA_DIR can point to a Render Persistent Disk (e.g. /data) so conversations
# survive service restarts.  Falls back to project root if not configured.
from .. import config as _cfg  # noqa: E402 (module-level import after Path setup)
_DATA_DIR       = Path(_cfg.DATA_DIR) if _cfg.DATA_DIR else _ROOT
_CONV_PATH      = _DATA_DIR / "conversations.json"
_LAST_SEEN_PATH = _DATA_DIR / "last_seen.json"

# Inactivity gap after which conversation history is reset (customer is treated as new)
_SESSION_GAP = 24 * 3600  # seconds

# ── System prompt — loaded from Supabase (fallback: file) ────────────────────
def _load_system_prompt_sync() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.critical("[BOOT] FATAL: Failed to load system prompt from file: %s", e)
        return ""

_SYSTEM_PROMPT: str = _load_system_prompt_sync()
logger.info("[BOOT] System prompt loaded from file (%d chars)", len(_SYSTEM_PROMPT))

async def _refresh_system_prompt() -> None:
    """Reload system prompt from file (source of truth) and sync to Supabase."""
    global _SYSTEM_PROMPT
    # 1. Always reload from file — the repo file is the source of truth
    fresh = _load_system_prompt_sync()
    if fresh:
        _SYSTEM_PROMPT = fresh
        DIAG_STATE["system_prompt_loaded"] = True
        DIAG_STATE["system_prompt_chars"] = len(fresh)
        logger.info("[RELOAD] System prompt reloaded from file (%d chars)", len(fresh))
        # 2. Push to Supabase so it stays in sync (for diagnostics / backup only)
        try:
            from ..providers.supabase_store import save_system_prompt
            ok = await save_system_prompt(fresh)
            if ok:
                logger.info("[SUPABASE] System prompt synced to Supabase")
        except Exception as e:
            logger.warning("[SUPABASE] Could not sync system prompt: %s", e)
    else:
        logger.warning("[RELOAD] System prompt file was empty — keeping previous version")

# ── FAQ bank — loaded from Supabase (fallback: file) ─────────────────────────
def _load_faq_sync() -> list:
    try:
        return json.loads(_FAQ_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("[BOOT] Failed to load FAQ bank from file: %s", e)
        return []

_faq_bank: list[dict] = _load_faq_sync()
logger.info("[BOOT] FAQ bank loaded from file (%d entries)", len(_faq_bank))

async def _refresh_faq() -> None:
    """Reload FAQ from file first, then try Supabase override."""
    global _faq_bank
    # Always reload from file
    fresh_file = _load_faq_sync()
    if fresh_file:
        _faq_bank = fresh_file
        DIAG_STATE["faq_count"] = len(fresh_file)
        logger.info("[RELOAD] FAQ bank reloaded from file (%d entries)", len(fresh_file))
    # Then try Supabase (may override with newer data)
    try:
        from ..providers.supabase_store import load_faq
        entries = await load_faq()
        if entries:
            _faq_bank = entries
            DIAG_STATE["faq_count"] = len(entries)
            logger.info("[SUPABASE] FAQ bank refreshed from Supabase (%d entries)", len(entries))
    except Exception as e:
        logger.warning("[SUPABASE] Could not refresh FAQ: %s", e)

# ── Fix 8: FAQ / system-prompt consistency check ─────────────────────────────
def _check_content_consistency() -> list[str]:
    """Compare key facts between system prompt and FAQ bank.
    Returns a list of discrepancy descriptions (empty = all consistent)."""
    issues: list[str] = []
    if not _SYSTEM_PROMPT or not _faq_bank:
        return issues

    # Phone numbers
    prompt_phones = set(re.findall(r'0\d{2}-\d{7}', _SYSTEM_PROMPT))
    faq_phones: set[str] = set()
    for entry in _faq_bank:
        faq_phones.update(re.findall(r'0\d{2}-\d{7}', entry.get("answer", "")))
    conflict = faq_phones - prompt_phones
    if conflict:
        issues.append(f"Phone number mismatch — FAQ contains {conflict} not in system prompt {prompt_phones}")

    # Showroom address (street + number)
    prompt_addr = set(re.findall(r'בעלי המלאכה\s+\d+', _SYSTEM_PROMPT))
    faq_addr: set[str] = set()
    for entry in _faq_bank:
        faq_addr.update(re.findall(r'בעלי המלאכה\s+\d+', entry.get("answer", "")))
    if faq_addr and prompt_addr and faq_addr != prompt_addr:
        issues.append(f"Address mismatch — FAQ: {faq_addr}, prompt: {prompt_addr}")

    # Friday closing hour
    prompt_fri = set(re.findall(r'ו[׳\']?\s+(?:עד\s+)?(\d{1,2}):00', _SYSTEM_PROMPT))
    faq_fri: set[str] = set()
    for entry in _faq_bank:
        faq_fri.update(re.findall(r'ו[׳\']?\s+(?:עד\s+)?(\d{1,2}):00', entry.get("answer", "")))
    if faq_fri and prompt_fri and faq_fri != prompt_fri:
        issues.append(f"Friday hours mismatch — FAQ: {faq_fri}:00, prompt: {prompt_fri}:00")

    return issues


_consistency_issues = _check_content_consistency()
for _issue in _consistency_issues:
    logger.critical("[CONSISTENCY] System prompt / FAQ mismatch: %s", _issue)
if not _consistency_issues and _faq_bank:
    logger.info("[BOOT] Content consistency check passed (%d FAQ entries)", len(_faq_bank))

# ── Diagnostics state (read by main.py /diag endpoint) ───────────────────────
DIAG_STATE: dict = {
    "system_prompt_loaded": bool(_SYSTEM_PROMPT),
    "system_prompt_chars":  len(_SYSTEM_PROMPT),
    "faq_count":            len(_faq_bank),
    "data_dir":             str(_DATA_DIR),
    "consistency_issues":   _consistency_issues,
    # AI provider tracking — updated on every _call_ai() invocation
    "ai_primary":           f"openrouter/openai/gpt-4.1-mini" if _cfg.OPENROUTER_API_KEY else "claude/claude-sonnet-4-6",
    "ai_fallback":          "claude/claude-sonnet-4-6" if _cfg.OPENROUTER_API_KEY else "none",
    "openrouter_key_set":   bool(_cfg.OPENROUTER_API_KEY),
    "last_ai_provider":     None,   # filled after first real AI call
    "openrouter_failures":  0,
    "last_openrouter_error": None,
}

# Max raw input length — truncated before hitting Claude to prevent abuse
_MAX_INPUT_CHARS = 2000

# ── Conversation history (in-memory, also persisted to disk) ──────────────────
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


# ── Per-sender structured conversation state ──────────────────────────────────
# Stores verified field values extracted from the conversation.
# Updated BEFORE every Claude call so Claude always sees the correct state.
# Never lost between turns — fields accumulate until the session resets.
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
    return {
        "full_name": None,
        "phone": None,
        "city": None,
        "preferred_contact_hours": None,
        "service_type": None,
        "project_status": None,
        "needs_frame_removal": None,
        "doors_count": None,
        "design_preference": None,
        "customer_gender_locked": None,
        "active_topics": [],
        "stage3_done": False,   # True once Stage 3 question has been sent AND answered
    }


# ── Known Israeli cities for city extraction ──────────────────────────────────
_ISRAELI_CITIES: set[str] = {
    # Primary service area
    "נתיבות", "באר שבע", "אשקלון", "אשדוד", "אופקים", "שדרות", "רהט", "דימונה",
    "קריית גת", "קריית מלאכי", "ערד", "אילת", "מצפה רמון", "ירוחם", "עומר",
    "להבים", "מיתר", "כסייפה", "חורה", "תל שבע", "רהט", "לקיה",
    # Major cities
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
    "ב\"ש", "ת\"א",   # shorthands
}

# Regex: Israeli phone numbers (0-prefix or 972-prefix, with optional separators)
_PHONE_RE = re.compile(
    r'(?<!\d)'
    r'(\+?972[-\s]?|0)'
    r'([5][0-9][-\s]?[0-9]{3}[-\s]?[0-9]{4}'
    r'|[5][0-9]{8})'
    r'(?!\d)'
)

# Hebrew word pattern (used for name validation)
_HEB_WORD_RE = re.compile(r'^[\u05d0-\u05fa][\u05d0-\u05fa\'\- ]{0,35}$')


def _extract_fields_from_message(text: str) -> dict:
    """
    Extract structured fields from a single customer message using regex.
    Returns only non-null values for fields that were confidently found.
    This runs BEFORE the Claude call so the extracted values are authoritative.
    """
    extracted: dict = {}
    t = text.strip()

    # ── Phone ─────────────────────────────────────────────────────────────────
    phone_match = _PHONE_RE.search(t)
    if phone_match:
        raw = re.sub(r'[-\s+]', '', phone_match.group(0))
        if raw.startswith('972'):
            raw = '0' + raw[3:]
        elif raw.startswith('+972'):
            raw = '0' + raw[4:]
        extracted['phone'] = raw

    # ── City ──────────────────────────────────────────────────────────────────
    for city in _ISRAELI_CITIES:
        if city in t:
            extracted['city'] = city
            break
    # Also detect "מאשקלון", "בנתיבות" etc. (preposition + city)
    if 'city' not in extracted:
        city_prep = re.search(r'(?:מ|ב|ל|ו)(נתיבות|באר שבע|אשקלון|אשדוד|אופקים|שדרות|ירושלים|תל אביב|חיפה|ראשון לציון|פתח תקווה|נתניה|רחובות)', t)
        if city_prep:
            extracted['city'] = city_prep.group(1)

    # ── Name — Hebrew word(s) adjacent to phone number ────────────────────────
    if phone_match:
        # Text before the phone number
        before = t[:phone_match.start()].strip()
        # Remove common name-introduction prefixes
        before = re.sub(r'^(?:שמי|קוראים לי|אני|שם שלי|השם שלי)\s*', '', before, flags=re.IGNORECASE).strip()
        # Valid name: 2–4 Hebrew words, no digits, not a city
        if (before
                and _HEB_WORD_RE.match(before)
                and before not in _ISRAELI_CITIES
                and len(before) >= 2):
            extracted['full_name'] = before

        if 'full_name' not in extracted:
            # Check after phone (e.g. "0523989366 ליטל")
            after = t[phone_match.end():].strip()
            # Strip city from after_phone so it doesn't get picked up as name
            if 'city' in extracted:
                after = after.replace(extracted['city'], '').strip()
            after = re.sub(r'^[,\s]+|[,\s]+$', '', after).strip()
            if (after
                    and _HEB_WORD_RE.match(after)
                    and after not in _ISRAELI_CITIES
                    and len(after) >= 2):
                extracted['full_name'] = after
    else:
        # No phone — look for explicit name markers
        name_m = re.match(
            r'^(?:שמי|קוראים לי|שם שלי|אני)\s+([\u05d0-\u05fa][\u05d0-\u05fa\'\- ]{1,30})',
            t, re.IGNORECASE
        )
        if name_m:
            candidate = name_m.group(1).strip()
            if candidate not in _ISRAELI_CITIES:
                extracted['full_name'] = candidate

    # ── Gender ────────────────────────────────────────────────────────────────
    if re.search(r'מחפשת|צריכה\b|מתעניינת|שמחה\b|מרוצה\b|מעוניינת|רציתי|קניתי\b|הגעתי\b', t):
        extracted['customer_gender_locked'] = 'female'
    elif re.search(r'מחפש\b|צריך\b|מתעניין\b|שמח\b|מעוניין\b', t):
        extracted['customer_gender_locked'] = 'male'

    # ── doors_count ───────────────────────────────────────────────────────────
    count_m = re.search(r'(\d+)\s*דלתות', t)
    if count_m:
        extracted['doors_count'] = int(count_m.group(1))

    # ── needs_frame_removal ───────────────────────────────────────────────────
    if re.search(r'כולל משקוף|עם משקוף|דלת ומשקוף', t, re.IGNORECASE):
        extracted['needs_frame_removal'] = True
    elif re.search(r'דלת בלבד|בלי משקוף|רק דלת\b|ללא משקוף|דלת לבד', t, re.IGNORECASE):
        extracted['needs_frame_removal'] = False

    # ── design_preference ─────────────────────────────────────────────────────
    if re.search(r'\bחלקה\b', t, re.IGNORECASE):
        extracted['design_preference'] = 'חלקה'
    elif re.search(r'\bמעוצבת\b', t, re.IGNORECASE):
        extracted['design_preference'] = 'מעוצבת'

    # ── project_status ────────────────────────────────────────────────────────
    if re.search(r'בית חדש|דירה חדשה|נכס חדש', t, re.IGNORECASE):
        extracted['project_status'] = 'בית חדש'
    elif re.search(r'\bשיפוץ\b|בשיפוץ\b|משפצים', t, re.IGNORECASE):
        extracted['project_status'] = 'שיפוץ'
    elif re.search(r'\bהחלפה\b|להחליף\b|דלת ישנה|קיימת', t, re.IGNORECASE):
        extracted['project_status'] = 'החלפה'

    # ── preferred_contact_hours ───────────────────────────────────────────────
    hours_m = re.search(r'אחרי\s*(\d{1,2})', t)
    if hours_m:
        h = int(hours_m.group(1))
        if h < 12:
            h += 12  # "אחרי 5" → 17:00
        extracted['preferred_contact_hours'] = f'אחרי {h:02d}:00'
    elif re.search(r'בוקר\b', t) and re.search(r'בין\s*(\d)', t):
        extracted['preferred_contact_hours'] = 'בבוקר'
    elif re.search(r'בכל שעה|בכל זמן|לא משנה', t, re.IGNORECASE):
        extracted['preferred_contact_hours'] = 'בכל שעה'

    return extracted


def _merge_state(existing: dict, new_fields: dict) -> dict:
    """
    Merge new_fields into the existing state dict.
    RULE: never overwrite a non-null/non-False value with null.
    gender_locked is set once and never changed.
    active_topics is a union (append-only).
    """
    merged = dict(existing)
    for key, value in new_fields.items():
        # Skip null/empty new values — existing wins
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        if key == 'active_topics':
            existing_list = merged.get('active_topics') or []
            if isinstance(value, list):
                for item in value:
                    if item not in existing_list:
                        existing_list.append(item)
            merged['active_topics'] = existing_list

        elif key == 'customer_gender_locked':
            # Lock once — never change after first detection
            if not merged.get('customer_gender_locked'):
                merged[key] = value

        elif key == 'needs_frame_removal':
            # False is a valid value — only skip if existing is already set
            if merged.get(key) is None:
                merged[key] = value

        else:
            # For all other fields: new value wins if it's non-null
            if merged.get(key) is None:
                merged[key] = value

    return merged


def _state_context_block(state: dict) -> str:
    """
    Build a context block injected into every Claude call.
    Shows Claude exactly which fields are already collected so it never re-asks them.
    """
    if not state:
        return ""

    def tick(val) -> str:
        if val is None:
            return "❌ NOT YET COLLECTED"
        if val is False:
            return "✅ false"
        if val is True:
            return "✅ true"
        return f"✅ {val}"

    phone = state.get('phone')
    name  = state.get('full_name')
    city  = state.get('city')
    hours = state.get('preferred_contact_hours')
    gender = state.get('customer_gender_locked')
    nfr   = state.get('needs_frame_removal')
    dp    = state.get('design_preference')
    dc    = state.get('doors_count')
    ps    = state.get('project_status')

    lines = [
        "## VERIFIED CONVERSATION STATE",
        "These values were extracted by the pre-processing layer. They are authoritative.",
        "⚠️  Do NOT ask the customer for any field marked ✅ — it is already known.",
        "",
        "Contact fields:",
        f"  phone:                   {tick(phone)}",
        f"  full_name:               {tick(name)}",
        f"  city:                    {tick(city)}",
        f"  preferred_contact_hours: {tick(hours)}",
        "",
        "Product fields:",
        f"  needs_frame_removal: {tick(nfr)}",
        f"  design_preference:   {tick(dp)}",
        f"  doors_count:         {tick(dc)}",
        f"  project_status:      {tick(ps)}",
        "",
    ]

    # Gender — explicit instruction
    if gender == 'female':
        lines.append("Gender: FEMALE — use לך / אלייך / תוכלי / תשאירי / תרצי in every reply.")
    elif gender == 'male':
        lines.append("Gender: MALE — use לך / אליך / תוכל / תשאיר / תרצה in every reply.")
    else:
        lines.append("Gender: UNKNOWN — use neutral plural: לכם / אליכם / תוכלו / תשאירו.")

    # Next contact field to ask
    lines.append("")
    if not phone:
        lines.append("NEXT CONTACT FIELD: phone — ask \"מה מספר הטלפון?\"")
    elif not name:
        lines.append("NEXT CONTACT FIELD: full_name — ask \"על שם מי הפנייה?\"")
    elif not city:
        lines.append("NEXT CONTACT FIELD: city — ask \"באיזו עיר מדובר?\"")
    else:
        lines.append("NEXT CONTACT FIELD: all collected — proceed to Stage 5 summary.")

    lines.append("")
    lines.append("If the customer's latest message contained contact info you haven't extracted above,")
    lines.append("extract ALL fields from it now, add them to your JSON output, and use the updated state.")

    return "\n".join(lines)


# ── Last-seen timestamps (tracks last customer message time per sender) ────────
# Used to detect 24h+ gaps and reset conversation to a clean slate.
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


# Senders explicitly reset via clear_conversation — next get_reply call starts fresh
# regardless of _last_seen or any residual disk state. In-memory only (intentional).
_force_fresh: set[str] = set()


# ── Business context ──────────────────────────────────────────────────────────
_BUSINESS = {
    "name":    "Michael Doors",
    "phone":   "054-2787578",
    "products": [
        "Entrance doors (smooth & designed): Nefachim, Panoramic, Greek, Mercury, Eden, Eden Brass series",
        "Interior doors: smooth, grooves, squares, arc, cross styles",
        "Hardware & handles: 15+ models, classic/hi-tech/synagogue styles",
        "Institutional & warehouse doors",
        "Synagogue doors (custom)",
    ],
    "hours": {"start": 9, "end": 18, "tz": "Asia/Jerusalem", "days": "א'–ה'", "fri_end": 13, "closed": "שבת וחגים"},
}


def is_working_hours() -> bool:
    """Return True when the business is currently open (exported for use in main.py)."""
    from datetime import datetime
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(_BUSINESS["hours"]["tz"]))
    hour, weekday = now.hour, now.weekday()  # 0=Mon … 5=Sat … 6=Sun (Israel: 4=Fri, 5=Sat)
    if weekday == 5:  # Saturday — closed
        return False
    if weekday == 4:  # Friday — half day
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
        f"Hours: Sun–Thu 09:00–18:00 | Fri 09:00–13:00 | Sat closed",
        f"Current time status: {status}",
    ])


# ── FAQ helpers ───────────────────────────────────────────────────────────────
def _find_faqs(user_msg: str) -> list[dict]:
    msg = user_msg.lower()
    matched = [e for e in _faq_bank if any(kw.lower() in msg for kw in e.get("keywords", []))]
    return matched[:3]


def _faq_block(faqs: list[dict]) -> str | None:
    if not faqs:
        return None
    lines = [f"[{f['category']}] {f['answer']}" for f in faqs]
    return "## מידע רלוונטי מבסיס הידע (לשימוש כהפניה בלבד — אל תעתיק את הניסוח)\n" + "\n".join(lines)


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
    """Return a human-readable Hebrew string for when the business next opens."""
    from datetime import datetime, timedelta
    import zoneinfo
    now = datetime.now(zoneinfo.ZoneInfo(_BUSINESS["hours"]["tz"]))
    wd = now.weekday()  # 0=Mon…4=Fri, 5=Sat, 6=Sun
    hour = now.hour
    h_start = _BUSINESS["hours"]["start"]
    h_end   = _BUSINESS["hours"]["end"]
    h_fri   = _BUSINESS["hours"]["fri_end"]

    # Still today (weekday, before closing)
    if wd < 4 and hour < h_end:
        return f"היום עד {h_end}:00"
    # Friday, still open
    if wd == 4 and hour < h_fri:
        return f"היום עד {h_fri}:00"
    # Friday after closing → Sunday
    if wd == 4 and hour >= h_fri:
        return f"ביום ראשון משעה {h_start}:00"
    # Saturday → Sunday
    if wd == 5:
        return f"ביום ראשון משעה {h_start}:00"
    # Sunday (wd=6) — open today if before h_end
    if wd == 6 and hour < h_end:
        return f"היום עד {h_end}:00"
    # Weekday evening (after closing) → tomorrow morning
    tomorrow_he = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
    next_day = tomorrow_he[(wd + 1) % 7]
    return f"ביום {next_day} משעה {h_start}:00"


def _build_system(user_msg: str, sender: str = "", state: dict | None = None) -> str:
    if not _SYSTEM_PROMPT:
        logger.error("System prompt is empty — Claude will have no instructions")
    greeting = _israel_greeting()
    parts = [
        _SYSTEM_PROMPT,
        f"## Business context\n{_context_block()}",
        (
            f"## Current time context\nCurrent greeting for this time of day: «{greeting}»\n"
            f"Use this greeting in reply_text_2 on the FIRST reply only — per RULE 1.\n"
            "On all subsequent replies (assistant message already exists in history): "
            "do NOT include any time-based greeting."
        ),
    ]
    # Inject verified conversation state so Claude never re-asks collected fields
    if state:
        state_block = _state_context_block(state)
        if state_block:
            parts.append(state_block)

    # Bypass hours check for test/admin phones
    _is_bypass = sender and sender in _cfg.HOURS_BYPASS_PHONES

    # Fix 7: explicit out-of-hours instruction injected on every call when closed
    if not is_working_hours() and not _is_bypass:
        next_open = _next_opening_time()
        parts.append(
            "## OUT-OF-HOURS — MANDATORY BEHAVIOUR\n"
            f"The business is currently CLOSED. Next opening: {next_open}.\n"
            "You MUST acknowledge this in your reply. Include ALL of the following:\n"
            f"1. We are not available right now but received the message.\n"
            f"2. We will call back {next_open}.\n"
            f"3. The customer can also call directly: 054-2787578.\n"
            "Still collect the 4 mandatory fields (שם, עיר, טלפון, שעה מועדפת) — "
            "a sales manager will review the lead in the morning.\n"
            "Do NOT promise immediate assistance."
        )

    # Fix 5: reinforce price policy on every single call — injected last so it's
    # not buried and harder for Claude to ignore
    parts.append(
        "## ABSOLUTE RULE — PRICE/DELIVERY DISCLOSURE FORBIDDEN\n"
        "NEVER state, estimate, hint at, or compare any price, price range, cost, or delivery time. "
        "This includes: specific amounts, 'around X', 'starting from X', 'up to X', "
        "'cheaper than', 'roughly X ₪', ranges like 'X–Y ₪', or spelled-out numbers. "
        "This rule overrides every other instruction in this prompt. "
        "If asked about price, respond with exactly: "
        "'המחיר מותאם אישית לפי סוג ועיצוב — מעולה! כדי שנוכל לחזור אליכם עם הצעת מחיר מסודרת, אשמח שתשאירו שם מלא, מספר טלפון ועיר מגורים 😊'"
    )

    faqs = _find_faqs(user_msg)
    block = _faq_block(faqs)
    if block:
        parts.append(block)
        logger.info("FAQ match: %s", ", ".join(f["id"] for f in faqs))
    return "\n\n".join(parts)


# ── Scenario classifier ───────────────────────────────────────────────────────
def _has_entrance(m: str)     -> bool: return bool(re.search(
    r"דלת כניסה|דלתות כניסה"
    r"|דלת חוץ|דלתות חוץ"                          # colloquial: "דלת חוץ" = entrance door
    r"|דלת חיצונית|דלתות חיצוניות"                 # formal synonym
    r"|דלת ראשית|דלתות ראשיות|וראשית\b|וכניסה\b"  # "main door" — also bare "וראשית" mid-sentence
    r"|דלת ברזל|דלת פלדה|דלתות ברזל|דלתות פלדה"   # iron/steel = typically entrance
    r"|כניסה לבית|כניסה לדירה|כניסה לבניין",       # "entrance to house/apartment/building"
    m))
def _has_interior(m: str)     -> bool: return bool(re.search(
    r"דלת פנים|דלתות פנים|פולימר"
    r"|דלת לחדר|דלת חדר|דלתות חדר"                 # "door to room"
    r"|דלת שירותים|דלת לשירותים"                   # bathroom door
    r"|דלת אמבטיה|דלת לאמבטיה"                    # bathroom door
    r"|דלת מטבח|דלת למטבח"                         # kitchen door
    r"|דלת שינה|דלת לשינה|דלת חדר שינה"           # bedroom door
    r"|דלת סלון|דלת לסלון",                        # living room door
    m))
def _has_specific_product(m: str) -> bool:
    """True when the message names a specific entrance door series or interior door material.
    Used in _detect_scenario to bypass scripted responses for entrance models only."""
    return bool(re.search(
        # Entrance door series (system prompt RULE 2)
        r"נפחות|נפחת|פנורמי|יווני|מרקורי|עדן|אומנויות|סביליה"
        # Interior door materials — these are NOT bypassed (see _detect_scenario)
        r"|פולימר|אלון|אגוז|MDF|HDF",
        m, re.IGNORECASE,
    ))
def _has_door_type(m: str)    -> bool: return _has_entrance(m) or _has_interior(m)
def _has_style(m: str)        -> bool: return bool(re.search(r"מודרנ|מעוצב|מעוצבת|קלאסי|קלאסית|חלקה|פשוטה", m))
def _is_question(m: str)      -> bool: return bool(re.search(r"\?|יש לכם|האם |אפשר ", m))
def _has_frame_removal(m: str) -> bool:
    return bool(re.search(r"פירוק משקוף|עם פירוק|בלי פירוק|להוציא משקוף|להחליף משקוף|ללא פירוק", m))
def _has_intent(m: str)       -> bool:
    return bool(re.search(
        r"מתעניין|מתעניינת|מעוניין|מעוניינת|רוצה|צריך|צריכה"
        r"|מחפש|מחפשת|מחפשים|מעניין אותי|מעניין אותנו|אשמח|מעוניינים"
        r"|להזמין|לקנות|לרכוש"                          # to order/buy/purchase
        r"|רוצה לדעת|מעניין אותו|מעניין אותה"           # want to know / he/she is interested
        r"|מעניין\b",                                    # standalone "interested"
        m))

def _is_greeting_only(m: str) -> bool:
    return bool(re.match(
        r"^(שלום|היי|הי|בוקר טוב|בוקר אור|ערב טוב|צהריים טובים|צהריים טוב|לילה טוב"
        r"|אהלן|טוב|מה שלומכם|מה נשמע|מה קורה|מה קורה שם|ספריד|חחח|אוקי|אחלה|נהדר|מצוין)[.!,\s]*$",
        m.strip(), re.IGNORECASE
    ))

_SCENARIOS: dict[str, dict] = {
    "greeting":                  {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Greeting only — pitch + asking how to help",                                          "response": _MSG["greeting"]},
    "showroom_address":          {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer asking for showroom address — gave address, collecting contact",            "response": _MSG["showroom_address"]},
    "showroom_hours":            {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer asking about hours — answered, offering to schedule visit",                 "response": _MSG["showroom_hours"]},
    "repair":                    {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer requesting repair — asking what the issue is",                              "response": _MSG["repair"]},
    "mamad":                     {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer asking about ממ\"ד door — clarifying new or existing",                      "response": _MSG["mamad"]},
    "frame_removal":             {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer asking about frame removal — door type still unknown",                      "response": _MSG["frame_removal"]},
    "designed_doors":            {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer interested in designed/modern doors — door type not specified",             "response": _MSG["designed_doors"]},
    "detailed_inquiry":          {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer specified entrance door with clear intent — asking about style next",       "response": _MSG["detailed_inquiry"]},
    "detailed_inquiry_interior": {"handoff_to_human": False, "needs_frame_removal": False, "summary": "Customer specified interior door with clear intent — asking for project status",     "response": _MSG["detailed_inquiry_interior"]},
    "entrance_doors":            {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer browsing entrance doors — guiding toward style preference",                 "response": _MSG["entrance_doors"]},
    "interior_doors":            {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer browsing interior doors — asking for project status",                       "response": _MSG["interior_doors"]},
    "vague_inquiry":             {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Quote request without door type — asking entrance vs interior",                      "response": _MSG["vague_inquiry"]},
    "ambiguous":                 {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Vague door inquiry — asking entrance vs interior",                                   "response": _MSG["ambiguous"]},
    "human_request":             {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Customer wants human agent — collecting name + phone",                               "response": _MSG["human_request"]},
    "emergency":                 {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Emergency — break-in or urgent door issue",                                          "response": _MSG["emergency"]},
    "contractor":                {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Contractor or large project — escalating to sales manager",                         "response": _MSG["contractor"]},
    "geographic":                {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Geographic coverage question — confirming service area",                             "response": _MSG["geographic"]},
    "sticker_only":              {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Sticker/emoji only — asking how to help",                                            "response": _MSG["sticker_only"]},
    "warranty":                  {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Warranty question — answered",                                                       "response": _MSG["warranty"]},
    "installation_time":         {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Installation time question — collecting contact for sales manager",                  "response": _MSG["installation_time"]},
    "colors":                    {"handoff_to_human": False, "needs_frame_removal": None,  "summary": "Colors question — answered with range",                                              "response": _MSG["colors"]},
}


def _detect_scenario(msg: str) -> dict | None:
    if _is_greeting_only(msg):
        return _SCENARIOS["greeting"]
    if re.search(r"כתובת|איפה אתם|איפה האולם|איפה החנות|המיקום שלכם|מיקום|להגיע אליכם|איך מגיעים", msg):
        return _SCENARIOS["showroom_address"]
    if re.search(r"שעות פעילות|שעות הפעילות|שעות פתיחה|שעות הפתיחה|מתי פתוח|מתי אתם פתוחים|מתי אפשר להגיע|לקבוע פגישה|תיאום פגישה|אולם תצוגה|אולם התצוגה|מה השעות|עד מתי פתוחים|מתי סגורים", msg):
        return _SCENARIOS["showroom_hours"]
    # Single emoji / sticker only
    if re.match(r"^[\U00010000-\U0010ffff\U00002600-\U000027BF\U0001F300-\U0001FAFF\s]+$", msg.strip()) and len(msg.strip()) <= 4:
        return _SCENARIOS["sticker_only"]
    # Emergency
    if re.search(r"פריצה|פרצו|שוד|חירום|עזרה דחופה|דחוף מאוד|דלת שבורה.*עכשיו|עכשיו.*דלת שבורה", msg):
        return _SCENARIOS["emergency"]
    # Human / agent request
    if re.search(r"נציג אנושי|נציג אמיתי|לדבר עם מישהו|לדבר עם אדם|לדבר עם נציג|שיחזרו אלי|תחזרו אלי|אדם אמיתי|בן אדם", msg):
        return _SCENARIOS["human_request"]
    # Contractor / large project
    if re.search(r"קבלן|פרויקט גדול|הרבה דלתות|עשרות דלתות|\d{2,} דלתות|בניין שלם|בניין חדש.*דלתות|דלתות.*בניין", msg):
        return _SCENARIOS["contractor"]
    # Geographic coverage
    if re.search(r"אתם מגיעים|אתם עובדים|מגיעים ל|תל אביב.*אתם|ירושלים.*אתם|חיפה.*אתם|אתם.*צפון|אתם.*מרכז|כיסוי גיאוגרפי|איזור השירות|עד איפה", msg):
        return _SCENARIOS["geographic"]
    # Warranty
    if re.search(r"אחריות|גארנטי|warranty|כמה שנים|כמה זמן אחריות|מה האחריות", msg):
        return _SCENARIOS["warranty"]
    # Installation time
    if re.search(r"כמה זמן לוקח|זמן התקנה|מתי יתקינו|תוך כמה זמן|זמן אספקה|מתי אפשר להתקין|כמה זמן עד|זמן המתנה", msg):
        return _SCENARIOS["installation_time"]
    # Colors
    if re.search(r"באיזה צבעים|איזה צבעים|צבעים יש|מה הצבעים|צבע.*דלת|דלת.*צבע|צבע אפשרי|גוונים", msg):
        return _SCENARIOS["colors"]
    if re.search(
        r"תיקון|תקלה|התקנתם|הותקנה|שירות לדלת"
        r"|בעיה בדלת|בעיה.*דלת|דלת.*בעיה"
        r"|הדלת לא נסגרת|הדלת לא נפתחת|הדלת תקועה|הדלת נפלה"
        r"|ציר שבור|ציר.*דלת|דלת.*ציר"              # hinge issues
        r"|מנעול שבור|בעיה במנעול|מנעול לא|לא נועל|לא נועלת"  # lock issues
        r"|ידית שבורה|ידית לא"                       # handle issues
        r"|התפרקה|התפרק|נשברה|נשבר"                 # came apart / broke
        r"|אחריות",
        msg,
    ):
        return _SCENARIOS["repair"]
    if re.search(r"ממ\"ד|ממד|דלת ממד|דלת ממ״ד|חדר ביטחון|מרחב מוגן", msg):
        return _SCENARIOS["mamad"]
    if _has_frame_removal(msg) and not _has_door_type(msg):
        return _SCENARIOS["frame_removal"]
    if _has_style(msg) and not _has_door_type(msg):
        return _SCENARIOS["designed_doors"]
    # Both door types in same message → Claude handles (greeting + proper dual response)
    if _has_entrance(msg) and _has_interior(msg):
        return None
    # Specific entrance-door model name (נפחות/פנורמי/יווני/מרקורי/עדן…) → Claude
    # handles with a model-specific reply per RULE 7 of the system prompt.
    # Interior door materials (פולימר, אלון, MDF…) are intentionally NOT bypassed —
    # they follow the normal interior door scenario flow below.
    if _has_specific_product(msg) and not _has_interior(msg):
        return None
    if _has_entrance(msg) and _has_intent(msg) and not _has_style(msg) and not _is_question(msg):
        return _SCENARIOS["detailed_inquiry"]
    if _has_interior(msg) and _has_intent(msg) and not _has_style(msg) and not _is_question(msg):
        return _SCENARIOS["detailed_inquiry_interior"]
    if _has_entrance(msg):
        return _SCENARIOS["entrance_doors"]
    if _has_interior(msg):
        return _SCENARIOS["interior_doors"]
    if re.search(
        r"הצעת מחיר|כמה עולה|כמה זה עולה|כמה עולים"
        r"|כמה יעלה|כמה יעלו|כמה זה יעלה|כמה יעלה לי|כמה יעלה לנו"  # future tense price
        r"|כמה לשים דלת|כמה לקנות דלת|כמה עולה לשים|כמה עולה להתקין"  # installation price
        r"|כמה זה|כמה ה|תעריף|מה העלות"             # colloquial "how much is it" / rates
        r"|מחיר|אפשר הצעה|מחיר.*דלת|דלת.*מחיר",
        msg,
    ):
        return _SCENARIOS["vague_inquiry"]
    if re.search(
        r"מחפש דלת|מחפשת דלת|מחפשים דלת|מתעניין|מתעניינת|מתעניינים"
        r"|מעוניין|מעוניינת|מעוניינים|דלת לבית|דלת לדירה|צריך דלת|צריכה דלת"
        r"|אשמח למידע|אפשר פרטים|פרטים על|רוצה לדעת|ספרו לי"
        r"|מה יש לכם|מה אתם מוכרים|מה אפשר|מה השירותים|מה המוצרים"
        r"|להזמין דלת|לקנות דלת|לרכוש דלת"          # order/buy a door
        r"|דלת לחנות|דלת למשרד|דלת למחסן|דלת למוסד"  # commercial doors
        r"|יש לכם דלת|האם יש לכם",                   # "do you have X"
        msg,
    ):
        return _SCENARIOS["ambiguous"]
    return None


# ── AI client (Claude or OpenRouter/GPT-4.1-mini) ─────────────────────────────
_claude: anthropic.AsyncAnthropic | None = None
_openrouter = None  # openai.AsyncOpenAI instance, created lazily

_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
_CLAUDE_MODEL     = "claude-sonnet-4-6"
_HAIKU_MODEL      = "claude-haiku-4-5-20251001"


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
    """Unified call — tries OpenRouter/GPT-4.1-mini first, falls back to Claude on any error."""
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
                DIAG_STATE["openrouter_failures"] = DIAG_STATE.get("openrouter_failures", 0)
                return content
            raise ValueError("OpenRouter returned empty content")
        except Exception as or_exc:
            logger.warning("[OPENROUTER:FAIL] %s — falling back to Claude", or_exc)
            DIAG_STATE["openrouter_failures"] = DIAG_STATE.get("openrouter_failures", 0) + 1
            DIAG_STATE["last_openrouter_error"] = str(or_exc)[:200]
            # fall through to Claude below

    # Claude (primary when no OpenRouter key, or fallback after OpenRouter failure)
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


_PARSE_ERROR_REPLY = _ERR["parse_error"]
_API_ERROR_REPLY   = _ERR["api_error"]

# Set of all error reply texts — used by main.py to skip follow-up after a failure
ERROR_REPLIES: frozenset[str] = frozenset([_PARSE_ERROR_REPLY, _API_ERROR_REPLY])

# Max conversation turns kept in memory (40 = ~20 back-and-forth exchanges)
_MAX_HISTORY = 40


# Matches explicit shekel amounts: "500 ₪", "₪1,200", "1500 שקל", "כ-3000 ש\"ח"
# Does NOT match "מחיר מותאם אישית" or similar non-numeric phrases.
_PRICE_RE = re.compile(
    r'(?:כ[-–]?|מ[-–]?|ב[-–]?|עד\s)?'
    r'\d[\d,\.]*\s*(?:₪|ש["\']?ח\b|שקל\b)'
    r'|(?:₪)\s*\d[\d,\.]*',
    re.UNICODE,
)


def _scrub_prices(text: str, sender: str) -> str:
    """Replace any explicit price amount with a safe placeholder."""
    if not _PRICE_RE.search(text):
        return text
    scrubbed = _PRICE_RE.sub("מחיר מותאם אישית", text)
    logger.warning("[PRICE:BLOCKED] Claude disclosed price — scrubbed | sender=%s | original=%s",
                   sender, text[:100])
    return scrubbed


def _extract_json(text: str) -> str:
    """Extract the last complete JSON object that contains 'reply_text'.
    Handles cases where the AI writes reasoning text before/inside the JSON."""
    # Strategy 1: find the LAST occurrence of {"reply_text": — the model's final answer
    marker = '"reply_text"'
    last_pos = text.rfind(marker)
    if last_pos > 0:
        # Walk backwards to find the opening {
        brace_pos = text.rfind("{", 0, last_pos)
        if brace_pos >= 0:
            candidate = text[brace_pos:]
            # Find the matching closing brace
            depth = 0
            for i, ch in enumerate(candidate):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return candidate[:i + 1]
    # Strategy 2: fall back to first { in text
    brace_pos = text.find("{")
    if brace_pos >= 0:
        return text[brace_pos:]
    return text


def _parse_response(raw: str, sender: str) -> dict:
    try:
        cleaned = raw.strip()
        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        # Strip bare "json" label that Claude sometimes adds without backticks
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE).strip()
        # Extract the last valid JSON object — handles AI "thinking" before/inside JSON
        cleaned = _extract_json(cleaned)
        parsed = json.loads(cleaned)
        reply_text = str(parsed.get("reply_text", ""))
        if not reply_text.strip():
            logger.warning("Empty reply_text in parsed response | sender=%s", sender)
            reply_text = _PARSE_ERROR_REPLY
        reply_text = _scrub_prices(reply_text, sender)
        reply_text_2_raw = parsed.get("reply_text_2")
        reply_text_2 = str(reply_text_2_raw).strip() if reply_text_2_raw else None
        if reply_text_2:
            reply_text_2 = _scrub_prices(reply_text_2, sender)
        return {
            "reply_text":              reply_text,
            "reply_text_2":            reply_text_2,
            "doors_count":             parsed.get("doors_count"),
            "handoff_to_human":        bool(parsed.get("handoff_to_human", False)),
            "summary":                 str(parsed.get("summary", "")),
            "preferred_contact_hours": parsed.get("preferred_contact_hours"),
            "needs_frame_removal":     parsed.get("needs_frame_removal"),
            "needs_installation":      parsed.get("needs_installation"),
            "full_name":               parsed.get("full_name"),
            "phone":                   parsed.get("phone"),
            "service_type":            parsed.get("service_type"),
            "city":                    parsed.get("city"),
            "design_preference":       parsed.get("design_preference"),
            "project_status":          parsed.get("project_status"),
            "referral_source":         parsed.get("referral_source"),
            "is_returning_customer":   parsed.get("is_returning_customer"),
        }
    except Exception:
        # Claude returned plain text instead of JSON — use it directly as the reply
        # rather than showing an error message to the customer.
        plain = raw.strip()
        if plain and plain not in (_PARSE_ERROR_REPLY, _API_ERROR_REPLY):
            logger.warning("Non-JSON response — using plain text | sender=%s | raw: %s", sender, raw[:120])
            plain = _scrub_prices(plain, sender)
            return {
                "reply_text": plain, "reply_text_2": None,
                "handoff_to_human": False,
                "summary": "Plain-text response (no JSON wrapper)",
                "preferred_contact_hours": None, "needs_frame_removal": None,
                "needs_installation": None, "full_name": None, "phone": None,
                "service_type": None, "city": None, "doors_count": None,
                "design_preference": None, "project_status": None,
                "referral_source": None, "is_returning_customer": None,
            }
        logger.warning("Non-JSON empty response | sender=%s — raw: %s", sender, raw[:120])
        return {
            "reply_text": _PARSE_ERROR_REPLY, "reply_text_2": None,
            "handoff_to_human": False,
            "summary": "Parse error — fallback reply sent",
            "preferred_contact_hours": None, "needs_frame_removal": None,
            "needs_installation": None, "full_name": None, "phone": None,
            "service_type": None, "city": None, "doors_count": None,
            "design_preference": None, "project_status": None,
            "referral_source": None, "is_returning_customer": None,
        }


def _sanitize_input(text: str, sender: str) -> str:
    """Truncate extreme-length input and strip control characters."""
    if len(text) > _MAX_INPUT_CHARS:
        logger.warning("[INPUT:TRUNCATE] Message truncated %d→%d | sender=%s", len(text), _MAX_INPUT_CHARS, sender)
        text = text[:_MAX_INPUT_CHARS]
    return text


def _validate_history(sender: str) -> None:
    """Remove any malformed entries from conversation history."""
    history = _conversations.get(sender, [])
    valid = [
        m for m in history
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]
    if len(valid) != len(history):
        logger.warning("[HIST:FIX] Removed %d malformed entries | sender=%s", len(history) - len(valid), sender)
        _conversations[sender] = valid


async def _supabase_save_conv(sender: str) -> None:
    try:
        from ..providers.supabase_store import save_conversation
        await save_conversation(sender, _conversations.get(sender, []))
    except Exception as e:
        logger.warning("[SUPABASE] save_conversation failed: %s", e)


async def get_reply(sender: str, user_message: str, anthropic_api_key: str, mock_claude: bool = False) -> dict:
    import time as _time
    user_message = _sanitize_input(user_message, sender)

    # ── Fresh-start gate ──────────────────────────────────────────────────────
    now = _time.time()
    if sender in _force_fresh:
        # Explicit #reset (or session close / handoff) — always wipe, works in both modes
        _force_fresh.discard(sender)
        _conversations.pop(sender, None)
        _conv_state.pop(sender, None)
        _last_seen.pop(sender, None)
        logger.info("[SESSION:FORCED] Fresh start after explicit reset | sender=%s", sender)
    elif not _cfg.TEST_MODE:
        # 24h inactivity auto-reset — production only.
        # In test mode only #reset can start a new conversation.
        last = _last_seen.get(sender, 0.0)
        if last > 0 and (now - last) > _SESSION_GAP:
            gap_h = (now - last) / 3600
            logger.info("[SESSION:RESET] %.1fh gap — fresh start | sender=%s", gap_h, sender)
            _conversations.pop(sender, None)
            _conv_state.pop(sender, None)
    _last_seen[sender] = now
    _save_last_seen()

    if sender not in _conversations:
        _conversations[sender] = []

    _conversations[sender].append({"role": "user", "content": user_message})
    if len(_conversations[sender]) > _MAX_HISTORY:
        _conversations[sender] = _conversations[sender][-_MAX_HISTORY:]
        logger.info("[HIST:TRIM] History trimmed to %d turns | sender=%s", _MAX_HISTORY, sender)

    _validate_history(sender)

    _COMPANY_PITCH = (
        "תודה שפניתם למיכאל דלתות 🚪\n"
        "אנו מציעים דלתות כניסה ופנים בסטנדרט הגבוה ביותר — התקנה, מכירה ואחריות מעל שנתיים ✨\n"
        "באולם התצוגה שלנו בנתיבות תוכלו להתרשם ממגוון רחב של דגמים 😊 (מומלץ לתאם פגישה מראש)"
    )

    # Scenario check on first message only.
    # Skip if the message looks like a mid-conversation answer (single short word
    # with no door-related noun) — likely the history was reset by a server restart.
    _looks_like_answer = bool(re.match(
        r"^(כן|לא|חלקה|מעוצבת|מודרנית|קלאסית|בית חדש|שיפוץ|החלפה|אולי|בסדר|אחלה|נכון|ברור|טוב)[.!?\s]*$",
        user_message.strip(), re.IGNORECASE
    ))
    if len(_conversations[sender]) == 1 and not _looks_like_answer:
        scenario = _detect_scenario(user_message)
        if scenario:
            logger.info("[SCENARIO] %s | sender=%s", scenario.get("summary", "?"), sender)
            greeting = _israel_greeting()
            # Pulse 1: company pitch (sent once per new conversation)
            msg1 = _COMPANY_PITCH
            # Pulse 2: actual response — strip legacy duplicate opening line if present,
            # then inject the time-based greeting via {{GREETING}} placeholder
            msg2 = re.sub(
                r"^היי, תודה שפניתם לדלתות מיכאל[^.\n]*\.?\n?",
                "",
                scenario["response"],
                count=1,
            ).strip()
            msg2 = msg2.replace("{{GREETING}}", greeting)
            # Store as one combined assistant turn so Claude sees clean history
            combined = msg1 + "\n\n" + msg2
            _conversations[sender].append({"role": "assistant", "content": combined})
            _save_conversations()
            asyncio.create_task(_supabase_save_conv(sender))
            return {
                "reply_text":              msg1,
                "reply_text_2":            msg2,
                "handoff_to_human":        scenario["handoff_to_human"],
                "summary":                 scenario["summary"],
                "preferred_contact_hours": None,
                "needs_frame_removal":     scenario["needs_frame_removal"],
                "needs_installation":      None,
                "full_name":               None,
                "phone":                   None,
                "service_type":            None,
                "city":                    None,
                "doors_count":             None,
                "design_preference":       None,
                "project_status":          None,
                "referral_source":         None,
                "is_returning_customer":   None,
            }

    # Mock mode — skip AI entirely (for UI testing without burning API credits)
    if mock_claude:
        turn = len(_conversations[sender])
        mock_reply = f"🤖 [מוק סיבוב {turn}] AI היה עונה כאן על: ״{user_message[:40]}״"
        _conversations[sender].append({"role": "assistant", "content": mock_reply})
        _save_conversations()
        return {
            "reply_text": mock_reply, "reply_text_2": None,
            "handoff_to_human": False,
            "summary": f"Mock mode — turn {turn}",
            "preferred_contact_hours": None, "needs_frame_removal": None,
            "needs_installation": None, "full_name": None, "phone": None,
            "service_type": None, "city": None, "doors_count": None,
            "design_preference": None,
            "project_status": None,
            "referral_source": None,
            "is_returning_customer": None,
        }

    # ── STATE PIPELINE (runs before every Claude call) ────────────────────────
    # Step 1: Initialise state for this sender if not yet exists
    if sender not in _conv_state:
        _conv_state[sender] = _empty_conv_state()

    # Step 2: Extract fields from the latest customer message using regex
    extracted = _extract_fields_from_message(user_message)
    if extracted:
        logger.info("[STATE:EXTRACT] sender=%s | extracted=%s", sender, extracted)

    # Step 3: Merge extracted fields into existing state (never overwrite non-null)
    _conv_state[sender] = _merge_state(_conv_state[sender], extracted)

    # Step 4: Save updated state to disk
    _save_conv_state()

    # Log the current known state for debugging
    s = _conv_state[sender]
    logger.info(
        "[STATE:CURRENT] sender=%s | phone=%s | name=%s | city=%s | gender=%s",
        sender, s.get('phone'), s.get('full_name'), s.get('city'), s.get('customer_gender_locked')
    )

    # ── AI call (OpenRouter/GPT-4.1-mini or Claude) ───────────────────────────
    provider = "openrouter" if _use_openrouter() else "claude"
    try:
        _t0 = _time.monotonic()
        logger.info("[AI:REQ] provider=%s | sender=%s | turns=%d", provider, sender, len(_conversations[sender]))
        raw_text = None
        for _attempt in range(3):
            try:
                raw_text = await _call_ai(
                    system=_build_system(user_message, sender, state=_conv_state[sender]),
                    messages=_conversations[sender],
                    max_tokens=900,
                    api_key=anthropic_api_key,
                    timeout=50.0,
                )
                break
            except (anthropic.RateLimitError, anthropic.APITimeoutError) as retry_exc:
                if _attempt == 2:
                    raise
                wait = 5 * (2 ** _attempt)
                logger.warning("[AI:RETRY] attempt=%d | sender=%s | %s — waiting %ds", _attempt + 1, sender, retry_exc, wait)
                await asyncio.sleep(wait)
        if not raw_text:
            raise ValueError("AI returned empty response")
        logger.info("[AI:OK] provider=%s | sender=%s | latency=%.1fs", provider, sender, _time.monotonic() - _t0)
    except Exception as exc:
        logger.error("[AI:ERR] provider=%s | sender=%s | type=%s | %s", provider, sender, type(exc).__name__, exc)
        fallback = _API_ERROR_REPLY
        _conversations[sender].append({"role": "assistant", "content": fallback})
        _save_conversations()
        return {
            "reply_text": fallback, "reply_text_2": None, "handoff_to_human": False,
            "summary": "AI API error — fallback sent",
            "preferred_contact_hours": None, "needs_frame_removal": None, "needs_installation": None,
        }

    structured = _parse_response(raw_text, sender)

    # ── Step 5: Merge Claude's JSON output back into state ────────────────────
    # Claude may have extracted additional fields we missed with regex (e.g. referral_source,
    # service_type, active_topics, stage3_done). Merge them in — our pre-extracted values
    # win for contact fields (phone/name/city) since they were already set above.
    claude_fields = {
        k: structured.get(k)
        for k in ('full_name', 'phone', 'city', 'preferred_contact_hours',
                  'service_type', 'project_status', 'needs_frame_removal',
                  'doors_count', 'design_preference', 'referral_source',
                  'is_returning_customer', 'needs_installation')
        if structured.get(k) is not None
    }
    # customer_gender_locked from Claude — only merge if not already locked
    claude_gender = structured.get('customer_gender_locked')
    if claude_gender and not _conv_state[sender].get('customer_gender_locked'):
        claude_fields['customer_gender_locked'] = claude_gender

    _conv_state[sender] = _merge_state(_conv_state[sender], claude_fields)
    _save_conv_state()

    # Patch the structured result to always reflect the authoritative state for contact fields
    for field in ('phone', 'full_name', 'city', 'preferred_contact_hours', 'customer_gender_locked'):
        if _conv_state[sender].get(field) is not None and structured.get(field) is None:
            structured[field] = _conv_state[sender][field]

    # Always strip greeting/pitch from Claude responses — these lines belong only
    # in the scripted first-pulse (sent separately). Strip from any position in the text.
    _GREETING_PAT = re.compile(
        r"(?:היי,?\s*)?תודה שפניתם (?:לדלתות מיכאל|למיכאל דלתות)[^\n]*\n?", re.IGNORECASE
    )
    _PITCH_PAT = re.compile(
        r"(?:אנחנו|אנו) מציעים דלתות כניסה ופנים[^\n]*\n?", re.IGNORECASE
    )

    def _strip_greeting(text: str) -> str:
        text = _GREETING_PAT.sub("", text).strip()
        text = _PITCH_PAT.sub("", text).strip()
        return text

    is_followup_turn = any(m["role"] == "assistant" for m in _conversations[sender][:-1])
    if is_followup_turn:
        stripped = _strip_greeting(structured["reply_text"])
        if stripped:
            structured["reply_text"] = stripped
        structured["reply_text_2"] = None
    elif structured.get("reply_text_2"):
        # First Claude turn: strip from pulse2 in case Claude duplicated the greeting there
        stripped2 = _strip_greeting(structured["reply_text_2"])
        structured["reply_text_2"] = stripped2 or None

    if structured["reply_text"] in ERROR_REPLIES:
        logger.warning("[FALLBACK] Parse fallback used | sender=%s | raw_preview=%s", sender, raw_text[:80])
    else:
        logger.info("[REPLY:OK] sender=%s | text=%s", sender, structured["reply_text"][:60])
    # Store both parts as a single assistant turn so history stays clean
    history_content = structured["reply_text"]
    if structured.get("reply_text_2"):
        history_content += "\n\n" + structured["reply_text_2"]
    _conversations[sender].append({"role": "assistant", "content": history_content})
    _save_conversations()
    asyncio.create_task(_supabase_save_conv(sender))
    return structured


async def get_followup_message(sender: str, anthropic_api_key: str) -> str:
    """15-min silence → personalized reminder that references the conversation topic."""
    history = _conversations.get(sender, [])
    _FALLBACK = "היי, עדיין ממתינה לתגובה מכם 😊 אם יש שאלה נוספת, אנחנו כאן לעזור!"
    if len(history) < 2:
        _conversations.setdefault(sender, []).append({"role": "assistant", "content": _FALLBACK})
        _save_conversations()
        return _FALLBACK
    system = (
        "אתה נציג מכירות של דלתות מיכאל. "
        "הלקוח לא ענה כבר 15 דקות. כתוב הודעת תזכורת קצרה בשורה אחת עד שתיים בסגנון הזה: "
        "\"היי, עדיין ממתינה לתגובה מכם 😊 אם יש עוד שאלות לגבי [נושא ספציפי מהשיחה], אנחנו כאן!\". "
        "החלף את [נושא ספציפי מהשיחה] בנושא האמיתי מהשיחה (סוג הדלת, השירות, הדגם שהוזכר). "
        "אם אין נושא ספציפי — השתמש בניסוח הגנרי: \"היי, עדיין ממתינה לתגובה מכם 😊 אם יש שאלה נוספת, אנחנו כאן לעזור!\". "
        "שפה ישירה ואנושית. בעברית בלבד. ללא JSON. ללא ברכות פתיחה נוספות."
    )
    try:
        msg = await _call_ai(system=system, messages=history[-6:], max_tokens=120, api_key=anthropic_api_key, timeout=15.0)
        msg = msg.strip()
        _conversations[sender].append({"role": "assistant", "content": msg})
        _save_conversations()
        return msg
    except Exception as exc:
        logger.error("get_followup_message error | sender=%s | %s", sender, exc)
        _conversations.setdefault(sender, []).append({"role": "assistant", "content": _FALLBACK})
        _save_conversations()
        return _FALLBACK


async def get_closing_message(sender: str, anthropic_api_key: str) -> str:
    """Generate a warm, personalized closing when customer says goodbye/thanks."""
    history = _conversations.get(sender, [])
    system = (
        "אתה נציג מכירות חם של דלתות מיכאל. "
        "הלקוח סיים את השיחה (אמר תודה / להתראות / הביע הסכמה). "
        "כתוב הודעת סגירה חמה וקצרה (2-3 שורות) שתכלול: "
        "1. תודה חמה על הפנייה עם אימוג'י אחד חיובי "
        "2. תזכורת שאפשר לחזור אלינו בכל עת "
        "3. פרטי יצירת קשר: 054-2787578 "
        "שפה אנושית, חמה, מכירתית. בעברית בלבד. ללא JSON."
    )
    try:
        msg = await _call_ai(
            system=system,
            messages=(history[-4:] if len(history) >= 4 else history),
            max_tokens=120,
            api_key=anthropic_api_key,
            timeout=15.0,
        )
        return msg.strip()
    except Exception as exc:
        logger.error("get_closing_message error | sender=%s | %s", sender, exc)
        return (
            "תודה רבה על הפנייה! 🙏\n"
            "אנחנו כאן בכל עת שתצטרכו — שיהיה לכם יום נהדר!\n"
            "דלתות מיכאל | 054-2787578"
        )


def is_closing_intent(message: str, conversation_turns: int) -> bool:
    """Return True if message looks like a farewell/thank-you and conversation is underway.
    Requires >= 4 turns so a lone 'תודה' on a 1-turn greeting doesn't trigger closing."""
    if conversation_turns < 4:
        return False
    m = message.strip()
    return bool(re.match(
        r"^(תודה רבה|תודה|תנקס|תנקיו|אחלה תודה|נשמע תודה|בסדר תודה|"
        r"אוקי תודה|אוקיי תודה|הבנתי תודה|קיבלתי תודה|"
        r"להתראות|ביי|שלום ולהתראות|יום טוב|שיהיה טוב|שיהיה יום טוב|"
        r"קיבלתי את המידע|"
        r"אצור קשר|אחזור אליך|נחזור|נצור קשר|אעלה בקשר)[.!,\s]*$",
        # Removed: הכל ברור / הכל טוב / הכל מובן — ambiguous, could be summary confirmation
        m, re.IGNORECASE,
    ))


async def generate_conversation_summary(sender: str, anthropic_api_key: str) -> str:
    """Generate a structured Hebrew summary of the full conversation for CRM use."""
    history = _conversations.get(sender, [])
    if not history:
        return "שיחה קצרה — לא נאסף מידע."
    system = (
        "אתה נציג של דלתות מיכאל. סכם את שיחת הוואטסאפ הבאה בעברית בצורה קצרה ומובנית.\n"
        "כלול את הסעיפים הרלוונטיים בלבד (אל תכתוב סעיף שאין לו מידע):\n"
        "• שם: [שם הלקוח אם ידוע]\n"
        "• עיר: [עיר מגורים אם ידועה]\n"
        "• בקשה: [מה הלקוח רצה — סוג דלת/שירות/דגם]\n"
        "• פרטים: [מידות, צבע, סגנון, פירוק משקוף — אם הוזכרו]\n"
        "• זמינות: [שעות מועדפות לחזרה אם הוזכרו]\n"
        "• סטטוס: [הועבר לנציג / נסגרה בידידות / נסגרה ללא מענה / בהמתנה]\n"
        "עד 6 שורות. ללא JSON. עברית בלבד."
    )
    try:
        result = await _call_ai(system=system, messages=history, max_tokens=200, api_key=anthropic_api_key, timeout=15.0)
        return result.strip()
    except Exception as exc:
        logger.error("generate_conversation_summary error | sender=%s | %s", sender, exc)
        return "שגיאה ביצירת סיכום."


def clear_conversation(sender: str) -> None:
    _conversations.pop(sender, None)
    _save_conversations()
    _conv_state.pop(sender, None)
    _save_conv_state()
    _last_seen.pop(sender, None)
    _save_last_seen()
    _force_fresh.add(sender)  # guarantees next message starts as a new conversation
    logger.info("Conversation cleared | %s", sender)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_supabase_delete_conv(sender))
    except RuntimeError:
        pass  # no running loop — deletion skipped (startup/test context)


async def _supabase_delete_conv(sender: str) -> None:
    try:
        from ..providers.supabase_store import delete_conversation
        await delete_conversation(sender)
    except Exception as e:
        logger.warning("[SUPABASE] delete_conversation failed: %s", e)
