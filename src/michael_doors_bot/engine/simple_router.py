"""
Scenario classifier + Claude service.
Scenario classifier runs only on the first message of a conversation.
All subsequent messages go directly to Claude.
"""
import json
import logging
import re
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent.parent.parent.parent
_PROMPT_PATH  = _ROOT / "src" / "prompts" / "systemPrompt.txt"
_FAQ_PATH     = _ROOT / "src" / "data" / "faqBank.json"
_CONV_PATH    = _ROOT / "conversations.json"

# ── FAQ bank (loaded once) ────────────────────────────────────────────────────
_faq_bank: list[dict] = json.loads(_FAQ_PATH.read_text(encoding="utf-8"))

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
    "hours": {"start": 8, "end": 18, "tz": "Asia/Jerusalem", "days": "א'–ה'", "closed": "שישי, שבת וחגים"},
}


def _is_working_hours() -> bool:
    from datetime import datetime
    import zoneinfo
    hour = datetime.now(zoneinfo.ZoneInfo(_BUSINESS["hours"]["tz"])).hour
    return _BUSINESS["hours"]["start"] <= hour < _BUSINESS["hours"]["end"]


def _context_block() -> str:
    status = (
        f"within working hours ({_BUSINESS['hours']['start']}:00–{_BUSINESS['hours']['end']}:00)"
        if _is_working_hours()
        else "outside working hours — let the customer know and offer to schedule a callback"
    )
    return "\n".join([
        f"Business: {_BUSINESS['name']}",
        f"Phone: {_BUSINESS['phone']}",
        f"Products: {', '.join(_BUSINESS['products'])}",
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


def _build_system(user_msg: str) -> str:
    parts = [
        _PROMPT_PATH.read_text(encoding="utf-8"),
        f"## Business context\n{_context_block()}",
        f"## Current time context\nCurrent Israeli greeting: {_israel_greeting()}. Use this greeting naturally when appropriate (e.g. in the opening message or when it fits the context). Do not force it into every message.",
    ]
    faqs = _find_faqs(user_msg)
    block = _faq_block(faqs)
    if block:
        parts.append(block)
        logger.info("FAQ match: %s", ", ".join(f["id"] for f in faqs))
    return "\n\n".join(parts)


# ── Scenario classifier ───────────────────────────────────────────────────────
def _has_entrance(m: str)     -> bool: return bool(re.search(r"דלת כניסה|דלתות כניסה", m))
def _has_interior(m: str)     -> bool: return bool(re.search(r"דלת פנים|דלתות פנים", m))
def _has_door_type(m: str)    -> bool: return _has_entrance(m) or _has_interior(m)
def _has_style(m: str)        -> bool: return bool(re.search(r"מודרנ|מעוצב|מעוצבת|קלאסי|קלאסית|חלקה|פשוטה", m))
def _is_question(m: str)      -> bool: return bool(re.search(r"\?|יש לכם|האם |אפשר ", m))
def _has_frame_removal(m: str) -> bool:
    return bool(re.search(r"פירוק משקוף|עם פירוק|בלי פירוק|להוציא משקוף|להחליף משקוף|ללא פירוק", m))
def _has_intent(m: str)       -> bool:
    return bool(re.search(r"מתעניין|מתעניינת|מעוניין|מעוניינת|רוצה|צריך|צריכה|מחפש|מחפשת|מחפשים|מעניין אותי|מעניין אותנו|אשמח|מעוניינים", m))

def _is_greeting_only(m: str) -> bool:
    return bool(re.match(
        r"^(שלום|היי|הי|בוקר טוב|ערב טוב|צהריים טובים|לילה טוב|אהלן|טוב|מה שלומכם|מה נשמע|ספריד|חחח|אוקי|אחלה|נהדר|מצוין)[.!,\s]*$",
        m.strip(), re.IGNORECASE
    ))

_SCENARIOS: dict[str, dict] = {
    "greeting": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Greeting only — asking how to help",
        "response": "היי, תודה שפניתם לדלתות מיכאל, איך אפשר לעזור?",
    },
    "showroom_address": {
        "handoff_to_human": True, "needs_frame_removal": None,
        "summary": "Customer asking for showroom address",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אולם התצוגה שלנו נמצא בבעלי המלאכה 15, נתיבות.\n"
            "שעות פעילות: ימים א'–ה', 08:00–18:00. סגור בימי שישי, שבת וחגים.\n"
            "מומלץ לתאם פגישת ייעוץ מראש כדי להבטיח שנציג יהיה פנוי לטפל בכם באופן מלא.\n"
            "אשמח שתשאירו שם מלא ומספר טלפון, וניצור איתכם קשר לתיאום."
        ),
    },
    "showroom_hours": {
        "handoff_to_human": True, "needs_frame_removal": None,
        "summary": "Customer asking about hours or scheduling a showroom visit",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "שעות הפעילות של אולם התצוגה: ימים א'–ה', 08:00–18:00. סגור בימי שישי, שבת וחגים.\n"
            "מומלץ לתאם פגישת ייעוץ מראש כדי להבטיח שנציג יהיה פנוי לטפל בכם באופן מלא.\n"
            "אשמח שתשאירו שם מלא, מספר טלפון ועיר מגורים, וניצור איתכם קשר לתיאום."
        ),
    },
    "repair": {
        "handoff_to_human": True, "needs_frame_removal": None,
        "summary": "Customer requesting repair or service",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אשמח שתכתבו שם מלא, עיר מגורים ומספר טלפון, וניצור איתכם קשר בהקדם."
        ),
    },
    "mamad": {
        "handoff_to_human": True, "needs_frame_removal": None,
        "summary": "Customer asking about ממ\"ד door — clarifying new or existing",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "האם מדובר בממ\"ד חדש או שיש ממ\"ד קיים שצריך להחליף את הדלת שלו?\n"
            "אשמח שתשאירו שם מלא, מספר טלפון ועיר מגורים, ונציג יחזור אליכם עם הצעת מחיר לאחר מדידה בשטח בהתאם למידות הפתח בממ\"ד."
        ),
    },
    "frame_removal": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer asking about frame removal — door type still unknown",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אשמח להבין האם מדובר בדלת כניסה או דלת פנים, כדי שנוכל לכוון אתכם נכון."
        ),
    },
    "designed_doors": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer interested in designed/modern doors — door type not specified",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "כן, בהחלט, יש אצלנו מגוון דלתות מעוצבות ודגמים בסגנונות שונים.\n"
            "אשמח להבין האם אתם מתעניינים בדלת כניסה או דלת פנים, כדי שנוכל לכוון אתכם נכון."
        ),
    },
    "detailed_inquiry": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer specified entrance door with clear intent — asking about style next",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "האם אתם מחפשים דלת חלקה או מעוצבת?"
        ),
    },
    "detailed_inquiry_interior": {
        "handoff_to_human": False, "needs_frame_removal": False,
        "summary": "Customer specified interior door with clear intent — asking for project status",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "כדי שנוכל להכין עבורכם הצעת מחיר מדויקת — האם מדובר בהחלפה של דלתות פנים קיימות, בית בשיפוץ, או שמדובר בבית חדש ללא דלתות פנים כרגע?"
        ),
    },
    "entrance_doors": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer browsing entrance doors — guiding toward style preference",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "כן, בהחלט, יש אצלנו מגוון דלתות כניסה בסגנונות שונים, כולל דגמים מודרניים ודלתות מעוצבות.\n"
            "אשמח שתכתבו אם אתם מחפשים דלת כניסה מודרנית, קלאסית או מעוצבת, ונכוון אתכם בהתאם."
        ),
    },
    "interior_doors": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer browsing interior doors — asking for project status",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "כן, בהחלט, יש אצלנו דלתות פנים במגוון סגנונות וצבעים.\n"
            "כדי שנוכל להכין עבורכם הצעת מחיר מדויקת — האם מדובר בהחלפה של דלתות פנים קיימות, בית בשיפוץ, או שמדובר בבית חדש ללא דלתות פנים כרגע?"
        ),
    },
    "vague_inquiry": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Quote request without door type — asking for clarification",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אשמח להבין באיזו דלת אתם מתעניינים — דלת כניסה או דלת פנים?"
        ),
    },
    "ambiguous": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Vague door inquiry — asking entrance vs interior",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אשמח להבין האם אתם מתעניינים בדלת כניסה או דלת פנים, כדי שנוכל לכוון אתכם נכון.\n"
            "אחרי שתכתבו את הפרט הזה, נמשיך יחד בצורה מסודרת."
        ),
    },
}


def _detect_scenario(msg: str) -> dict | None:
    if _is_greeting_only(msg):
        return _SCENARIOS["greeting"]
    if re.search(r"כתובת|איפה אתם|איפה האולם|איפה החנות|המיקום שלכם|מיקום|להגיע אליכם|איך מגיעים", msg):
        return _SCENARIOS["showroom_address"]
    if re.search(r"שעות פעילות|שעות הפעילות|שעות פתיחה|שעות הפתיחה|מתי פתוח|מתי אפשר להגיע|לקבוע פגישה|תיאום פגישה|אולם תצוגה|אולם התצוגה", msg):
        return _SCENARIOS["showroom_hours"]
    if re.search(r"תיקון|תקלה|התקנתם|הותקנה|שירות לדלת|בעיה בדלת|בעיה.*דלת|דלת.*בעיה|הדלת לא נסגרת|הדלת לא נפתחת|ציר שבור|ידית שבורה|אחריות", msg):
        return _SCENARIOS["repair"]
    if re.search(r"ממ\"ד|ממד|דלת ממד|דלת ממ״ד|חדר ביטחון|מרחב מוגן", msg):
        return _SCENARIOS["mamad"]
    if _has_frame_removal(msg) and not _has_door_type(msg):
        return _SCENARIOS["frame_removal"]
    if _has_style(msg) and not _has_door_type(msg):
        return _SCENARIOS["designed_doors"]
    if _has_entrance(msg) and _has_intent(msg) and not _has_style(msg) and not _is_question(msg):
        if _has_style(msg):
            return {**_SCENARIOS["detailed_inquiry"],
                    "summary": "Customer specified entrance door + style — asking about frame removal",
                    "response": "היי, תודה שפניתם לדלתות מיכאל.\nהאם יש צורך בהחלפת משקוף קיים?"}
        return _SCENARIOS["detailed_inquiry"]
    if _has_interior(msg) and _has_intent(msg) and not _has_style(msg) and not _is_question(msg):
        return _SCENARIOS["detailed_inquiry_interior"]
    if _has_entrance(msg):
        return _SCENARIOS["entrance_doors"]
    if _has_interior(msg):
        return _SCENARIOS["interior_doors"]
    if re.search(r"הצעת מחיר|כמה עולה|כמה זה עולה|כמה עולים|מחיר|אפשר הצעה|מחיר.*דלת|דלת.*מחיר", msg):
        return _SCENARIOS["vague_inquiry"]
    if re.search(r"מחפש דלת|מחפשת דלת|מחפשים דלת|מתעניין|מתעניינת|מתעניינים|מעוניין|מעוניינת|מעוניינים|דלת לבית|דלת לדירה|צריך דלת|צריכה דלת|אשמח למידע|אפשר פרטים|פרטים על|רוצה לדעת|ספרו לי|מה יש לכם|מה אתם מוכרים|מה אפשר|מה השירותים|מה המוצרים", msg):
        return _SCENARIOS["ambiguous"]
    return None


# ── Claude client ─────────────────────────────────────────────────────────────
_claude: anthropic.AsyncAnthropic | None = None


def _get_claude(api_key: str) -> anthropic.AsyncAnthropic:
    global _claude
    if _claude is None:
        _claude = anthropic.AsyncAnthropic(api_key=api_key)
    return _claude


def _parse_response(raw: str, sender: str) -> dict:
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        return {
            "reply_text":              str(parsed.get("reply_text", "")),
            "handoff_to_human":        bool(parsed.get("handoff_to_human", False)),
            "summary":                 str(parsed.get("summary", "")),
            "preferred_contact_hours": parsed.get("preferred_contact_hours"),
            "needs_frame_removal":     parsed.get("needs_frame_removal"),
            "needs_installation":      parsed.get("needs_installation"),
            "full_name":               parsed.get("full_name"),
            "service_type":            parsed.get("service_type"),
            "city":                    parsed.get("city"),
        }
    except Exception:
        logger.warning("Non-JSON response | sender=%s — using raw text", sender)
        return {
            "reply_text": raw, "handoff_to_human": False,
            "summary": "Parse error — raw reply returned",
            "preferred_contact_hours": None, "needs_frame_removal": None,
            "needs_installation": None, "full_name": None, "service_type": None,
            "city": None,
        }


async def get_reply(sender: str, user_message: str, anthropic_api_key: str) -> dict:
    if sender not in _conversations:
        _conversations[sender] = []

    _conversations[sender].append({"role": "user", "content": user_message})
    if len(_conversations[sender]) > 20:
        _conversations[sender] = _conversations[sender][-20:]

    # Scenario check on first message only
    if len(_conversations[sender]) == 1:
        scenario = _detect_scenario(user_message)
        if scenario:
            logger.info("Scenario: %s | %s", scenario.get("summary", "?"), sender)
            _conversations[sender].append({"role": "assistant", "content": scenario["response"]})
            _save_conversations()
            return {
                "reply_text":              scenario["response"],
                "handoff_to_human":        scenario["handoff_to_human"],
                "summary":                 scenario["summary"],
                "preferred_contact_hours": None,
                "needs_frame_removal":     scenario["needs_frame_removal"],
                "needs_installation":      None,
            }

    # Claude
    try:
        logger.info("Claude request | sender=%s | turns=%d", sender, len(_conversations[sender]))
        client = _get_claude(anthropic_api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=_build_system(user_message),
            messages=_conversations[sender],
            timeout=50.0,
        )
        raw_text = response.content[0].text
    except Exception as exc:
        logger.error("Claude API error | sender=%s | %s", sender, exc)
        fallback = "מצטערים, אירעה תקלה זמנית. אנא נסו שנית בעוד רגע."
        _conversations[sender].append({"role": "assistant", "content": fallback})
        _save_conversations()
        return {
            "reply_text": fallback, "handoff_to_human": False,
            "summary": "Claude API error — fallback sent",
            "preferred_contact_hours": None, "needs_frame_removal": None, "needs_installation": None,
        }

    structured = _parse_response(raw_text, sender)
    logger.info("Reply created | sender=%s | text=%s", sender, structured["reply_text"][:60])
    _conversations[sender].append({"role": "assistant", "content": structured["reply_text"]})
    _save_conversations()
    return structured


async def get_followup_message(sender: str, anthropic_api_key: str) -> str:
    history = _conversations.get(sender, [])
    if len(history) < 2:
        return (
            "שלום, רצינו לוודא שהמידע שמסרנו היה ברור. "
            "האם תרצו להמשיך את הפנייה, או שנסגור אותה בינתיים?"
        )
    client = _get_claude(anthropic_api_key)
    system = (
        "אתה נציג מכירות של דלתות מיכאל. "
        "כתוב הודעת המשך קצרה ומקצועית (עד 3 שורות) ללקוח שלא ענה 15 דקות. "
        "ההודעה צריכה: לציין שהפנייה עדיין פתוחה, לחדש בקצרה את הנושא שנדון, "
        "ולשאול אם הלקוח רוצה להמשיך את השיחה או לסגור את הפנייה. "
        "בעברית בלבד. ללא JSON. ללא אימוג'ים. שפה חמה ומקצועית."
    )
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system,
            messages=history[-6:],
            timeout=15.0,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("get_followup_message error | sender=%s | %s", sender, exc)
        return (
            "שלום, רצינו לוודא שהמידע שמסרנו היה ברור. "
            "האם תרצו להמשיך את הפנייה, או שנסגור אותה בינתיים?"
        )


def clear_conversation(sender: str) -> None:
    if sender in _conversations:
        del _conversations[sender]
        _save_conversations()
        logger.info("Conversation cleared | %s", sender)
