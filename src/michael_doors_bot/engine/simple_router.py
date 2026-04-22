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
    "hours": {"start": 9, "end": 18, "tz": "Asia/Jerusalem", "days": "א'–ה'", "fri_end": 13, "closed": "שבת וחגים"},
}


def _is_working_hours() -> bool:
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
            if _is_working_hours()
            else f"outside working hours (Friday closes at {_BUSINESS['hours']['fri_end']}:00)"
        )
    else:
        status = (
            f"within working hours ({_BUSINESS['hours']['start']}:00–{_BUSINESS['hours']['end']}:00)"
            if _is_working_hours()
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


def _build_system(user_msg: str) -> str:
    parts = [
        _PROMPT_PATH.read_text(encoding="utf-8"),
        f"## Business context\n{_context_block()}",
        f"## Current time context\nGreeting to use: «{_israel_greeting()}»\nCRITICAL: If there is NO prior assistant message in the conversation history — this is the first reply. You MUST embed the greeting inside the opening line, like this: 'היי, תודה שפניתם לדלתות מיכאל, {_israel_greeting()} 😊'. Never skip this on a first reply. Never repeat it after the first reply.",
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
        "summary": "Greeting only — pitch + asking how to help",
        "response": "היי, תודה שפניתם לדלתות מיכאל.\nבמה אפשר לעזור?",
    },
    "showroom_address": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer asking for showroom address — gave address, collecting contact",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אולם התצוגה שלנו נמצא בבעלי המלאכה 15, נתיבות 📍\n"
            "שעות פעילות: א'–ה' 09:00–18:00 | ו' 09:00–13:00 | שבת סגור.\n"
            "מומלץ לתאם פגישת ייעוץ מראש — רוצים שאקבע לכם זמן?"
        ),
    },
    "showroom_hours": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer asking about hours — answered, offering to schedule visit",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "שעות הפעילות: א'–ה' 09:00–18:00 | ו' 09:00–13:00 | שבת סגור 😊\n"
            "רוצים לתאם ביקור באולם התצוגה?"
        ),
    },
    "repair": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer requesting repair — asking what the issue is",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "אוי, אני ממש מצטערת לשמוע 💙 בואו נטפל בזה מיד.\n"
            "מה בדיוק קורה עם הדלת?"
        ),
    },
    "mamad": {
        "handoff_to_human": False, "needs_frame_removal": None,
        "summary": "Customer asking about ממ\"ד door — clarifying new or existing",
        "response": (
            "היי, תודה שפניתם לדלתות מיכאל.\n"
            "שמחה לעזור! האם מדובר בממ\"ד חדש, או שיש ממ\"ד קיים שרוצים להחליף את דלתו?"
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
    # Both door types in same message → Claude handles (greeting + proper dual response)
    if _has_entrance(msg) and _has_interior(msg):
        return None
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


_PARSE_ERROR_REPLY = "מצטערים, אירעה תקלה זמנית. אנא נסו שנית בעוד רגע 🙏"
_API_ERROR_REPLY   = "מצטערים, אירעה תקלה זמנית. אנא נסו שנית בעוד רגע."

# Set of all error reply texts — used by main.py to skip follow-up after a failure
ERROR_REPLIES: frozenset[str] = frozenset([_PARSE_ERROR_REPLY, _API_ERROR_REPLY])


def _parse_response(raw: str, sender: str) -> dict:
    try:
        cleaned = raw.strip()
        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        # Strip bare "json" label that Claude sometimes adds without backticks
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE).strip()
        parsed = json.loads(cleaned)
        return {
            "reply_text":              str(parsed.get("reply_text", "")),
            "handoff_to_human":        bool(parsed.get("handoff_to_human", False)),
            "summary":                 str(parsed.get("summary", "")),
            "preferred_contact_hours": parsed.get("preferred_contact_hours"),
            "needs_frame_removal":     parsed.get("needs_frame_removal"),
            "needs_installation":      parsed.get("needs_installation"),
            "full_name":               parsed.get("full_name"),
            "phone":                   parsed.get("phone"),
            "service_type":            parsed.get("service_type"),
            "city":                    parsed.get("city"),
        }
    except Exception:
        logger.warning("Non-JSON response | sender=%s — raw: %s", sender, raw[:120])
        return {
            "reply_text": _PARSE_ERROR_REPLY, "handoff_to_human": False,
            "summary": "Parse error — fallback reply sent",
            "preferred_contact_hours": None, "needs_frame_removal": None,
            "needs_installation": None, "full_name": None, "phone": None,
            "service_type": None, "city": None,
        }


async def get_reply(sender: str, user_message: str, anthropic_api_key: str) -> dict:
    if sender not in _conversations:
        _conversations[sender] = []

    _conversations[sender].append({"role": "user", "content": user_message})
    if len(_conversations[sender]) > 20:
        _conversations[sender] = _conversations[sender][-20:]

    _COMPANY_PITCH = (
        "אנחנו מציעים דלתות כניסה ופנים באיכות הגבוהה ביותר בשוק — "
        "מגוון רחב של דגמים ועיצובים בהתאמה אישית, אחריות מקיפה של מעל שנתיים, "
        "ואולם תצוגה מרשים בנתיבות שבו תוכלו להתרשם ולמצוא בדיוק את מה שמתאים לבית שלכם. 🚪✨"
    )

    # Scenario check on first message only
    if len(_conversations[sender]) == 1:
        scenario = _detect_scenario(user_message)
        if scenario:
            logger.info("Scenario: %s | %s", scenario.get("summary", "?"), sender)
            # Inject greeting + company pitch into the opening line
            greeting = _israel_greeting()
            response = scenario["response"].replace(
                "היי, תודה שפניתם לדלתות מיכאל",
                f"היי, תודה שפניתם לדלתות מיכאל, {greeting} 😊\n{_COMPANY_PITCH}",
                1,
            )
            _conversations[sender].append({"role": "assistant", "content": response})
            _save_conversations()
            return {
                "reply_text":              response,
                "handoff_to_human":        scenario["handoff_to_human"],
                "summary":                 scenario["summary"],
                "preferred_contact_hours": None,
                "needs_frame_removal":     scenario["needs_frame_removal"],
                "needs_installation":      None,
                "full_name":               None,
                "service_type":            None,
                "city":                    None,
            }

    # Claude
    try:
        logger.info("Claude request | sender=%s | turns=%d", sender, len(_conversations[sender]))
        client = _get_claude(anthropic_api_key)
        # Prefill assistant turn with "{" — forces Claude to output valid JSON
        # and makes it impossible for any label/code-fence to appear before the object.
        prefilled_messages = _conversations[sender] + [{"role": "assistant", "content": "{"}]
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            system=_build_system(user_message),
            messages=prefilled_messages,
            timeout=50.0,
        )
        raw_text = "{" + response.content[0].text
    except Exception as exc:
        logger.error("Claude API error | sender=%s | %s", sender, exc)
        fallback = _API_ERROR_REPLY
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
    """15-min silence → personalized reminder that references the conversation topic."""
    history = _conversations.get(sender, [])
    if len(history) < 2:
        return (
            "היי, רק רצינו לוודא שקיבלת את כל המידע שצריך 😊\n"
            "הפנייה עדיין פתוחה — אם יש שאלה נוספת או רוצה להמשיך, נשמח לעזור!"
        )
    client = _get_claude(anthropic_api_key)
    system = (
        "אתה נציג מכירות חם ומקצועי של דלתות מיכאל. "
        "הלקוח לא ענה 15 דקות. כתוב הודעת תזכורת קצרה (2-3 שורות) שתכלול: "
        "1. אזכור ספציפי של נושא השיחה (איזה דלת/שירות הלקוח שאל עליו) "
        "2. הצעה להמשיך — שאלה קצרה שתעודד תגובה "
        "3. אימוג'י אחד בלבד שמתאים להקשר "
        "שפה אנושית וחמה, לא רובוטית. בעברית בלבד. ללא JSON."
    )
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=system,
            messages=history[-6:],
            timeout=15.0,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("get_followup_message error | sender=%s | %s", sender, exc)
        return (
            "היי, רק רצינו לוודא שקיבלת את כל המידע שצריך 😊\n"
            "הפנייה עדיין פתוחה — אם יש שאלה נוספת או רוצה להמשיך, נשמח לעזור!"
        )


async def get_closing_message(sender: str, anthropic_api_key: str) -> str:
    """Generate a warm, personalized closing when customer says goodbye/thanks."""
    history = _conversations.get(sender, [])
    client = _get_claude(anthropic_api_key)
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
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=system,
            messages=(history[-4:] if len(history) >= 4 else history),
            timeout=15.0,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("get_closing_message error | sender=%s | %s", sender, exc)
        return (
            "תודה רבה על הפנייה! 🙏\n"
            "אנחנו כאן בכל עת שתצטרכו — שיהיה לכם יום נהדר!\n"
            "דלתות מיכאל | 054-2787578"
        )


def is_closing_intent(message: str, conversation_turns: int) -> bool:
    """Return True if message looks like a farewell/thank-you and conversation is underway."""
    if conversation_turns < 2:
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
    client = _get_claude(anthropic_api_key)
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
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system,
            messages=history,
            timeout=15.0,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.error("generate_conversation_summary error | sender=%s | %s", sender, exc)
        return "שגיאה ביצירת סיכום."


def clear_conversation(sender: str) -> None:
    if sender in _conversations:
        del _conversations[sender]
        _save_conversations()
        logger.info("Conversation cleared | %s", sender)
