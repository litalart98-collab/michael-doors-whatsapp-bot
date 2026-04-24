"""
Automated test scenarios for Michael Doors bot.
Each scenario is a conversation flow with assertions on the final state.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

# ── Scenario definition ───────────────────────────────────────────────────────

@dataclass
class Message:
    text: str
    # Assertions checked after THIS message's response:
    expect_handoff: bool | None = None          # True/False/None=don't check
    expect_service_type: str | None = None      # substring match
    expect_doors_count: int | None = None
    expect_no_price: bool = False               # response must not contain shekel/price
    expect_text_contains: str | None = None     # substring in reply_text
    expect_text_not_contains: str | None = None

@dataclass
class Scenario:
    id: str
    name: str
    messages: list[Message]
    description: str = ""

# ── All scenarios ─────────────────────────────────────────────────────────────

SCENARIOS: list[Scenario] = [

    Scenario(
        id="greeting",
        name="פתיחת שיחה — ברכה בלבד",
        description="לקוח שולח שלום — הבוט מברך ומציג את החברה",
        messages=[
            Message("שלום",
                    expect_handoff=False,
                    expect_text_contains="מיכאל"),
        ],
    ),

    Scenario(
        id="price_blocked",
        name="חסימת מחיר",
        description="הבוט לא מספק מחיר בשום צורה",
        messages=[
            Message("כמה עולה דלת כניסה?",
                    expect_no_price=True,
                    expect_handoff=False),
        ],
    ),

    Scenario(
        id="showroom_address",
        name="כתובת אולם תצוגה",
        description="הבוט נותן כתובת נכונה של הנציבות",
        messages=[
            Message("איפה האולם תצוגה שלכם?",
                    expect_text_contains="נתיבות",
                    expect_handoff=False),
        ],
    ),

    Scenario(
        id="human_request",
        name="בקשה לנציג אנושי",
        description="לקוח מבקש נציג — handoff מהיר אחרי שם וטלפון",
        messages=[
            Message("אני רוצה לדבר עם נציג"),
            Message("דוד לוי"),
            Message("0521234567",
                    expect_handoff=True),
        ],
    ),

    Scenario(
        id="repair_request",
        name="בקשת תיקון / אחריות",
        description="לקוח עם בעיה בדלת — הבוט מגלה אמפתיה ואוסף פרטים",
        messages=[
            Message("הדלת שלי לא נסגרת טוב",
                    expect_handoff=False,
                    expect_text_not_contains="₪"),
            Message("אשקלון"),
            Message("רחל כהן"),
            Message("0509876543"),
            Message("אחה\"צ",
                    expect_handoff=True,
                    expect_service_type="תיקון"),
        ],
    ),

    Scenario(
        id="entrance_door_full",
        name="דלת כניסה — זרימה מלאה עד handoff",
        description="לקוח מתעניין בדלת כניסה, עונה על כל השאלות",
        messages=[
            Message("אני מחפש דלת כניסה לדירה שלי",
                    expect_handoff=False),
            Message("יש משקוף קיים",
                    expect_handoff=False),
            Message("אני רוצה משהו מודרני",
                    expect_handoff=False),
            Message("באר שבע"),
            Message("יוסי לוי"),
            Message("0521111111"),
            Message("בבוקר בין 9 ל-11",
                    expect_handoff=True,
                    expect_service_type="כניסה"),
        ],
    ),

    Scenario(
        id="interior_doors_count",
        name="דלתות פנים — כמות",
        description="לקוח מציין כמות — doors_count נקלט נכון",
        messages=[
            Message("אני צריך 4 דלתות פנים לדירה",
                    expect_doors_count=4,
                    expect_service_type="פנים"),
        ],
    ),

    Scenario(
        id="mamad_door",
        name="דלת ממ\"ד",
        description="הבוט מסביר על תקן ת\"י 5044 ואוסף פרטים",
        messages=[
            Message("אני צריך דלת ממד",
                    expect_handoff=False,
                    expect_service_type="ממ"),
        ],
    ),

    Scenario(
        id="multi_intent",
        name="כוונה כפולה — כניסה + פנים",
        description="לקוח מתעניין בשני סוגי דלתות — הבוט מטפל בשניהם",
        messages=[
            Message("אני מחפש גם דלת כניסה וגם דלתות פנים",
                    expect_handoff=False),
        ],
    ),

    Scenario(
        id="sticker_handling",
        name="טיפול בסטיקר/תמונה",
        description="הבוט מגיב בחן להודעה לא טקסטואלית",
        messages=[
            Message("[סטיקר]",
                    expect_handoff=False),
        ],
    ),

    Scenario(
        id="short_answer_flow",
        name="תשובות קצרות — זרימה רציפה",
        description="הבוט מקבל תשובות חד-מילה ומתקדם בלי לחזור על שאלות",
        messages=[
            Message("דלת פנים"),
            Message("3"),
            Message("מודרני"),
            Message("חיפה"),
            Message("אמיר"),
            Message("0541234567"),
            Message("ערב",
                    expect_handoff=True),
        ],
    ),

    Scenario(
        id="geographic_coverage",
        name="איזור גיאוגרפי — צפון הארץ",
        description="הבוט מאשר שירות בכל הארץ",
        messages=[
            Message("אתם מגיעים לחיפה?",
                    expect_handoff=False,
                    expect_text_not_contains="לא מגיעים"),
        ],
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    turn: int
    message: str
    reply: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

@dataclass
class ScenarioResult:
    scenario: Scenario
    steps: list[StepResult] = field(default_factory=list)
    passed: bool = True
    duration_ms: int = 0
    error: str | None = None


_PRICE_PATTERNS = ["₪", "שקל", "עולה", "מחיר", "עלות", "תמחור"]


def _check_message(msg: Message, result: dict, reply: str) -> list[str]:
    failures = []

    if msg.expect_handoff is not None:
        got = bool(result.get("handoff_to_human"))
        if got != msg.expect_handoff:
            failures.append(
                f"handoff_to_human={got} (expected {msg.expect_handoff})"
            )

    if msg.expect_service_type is not None:
        svc = (result.get("service_type") or "").lower()
        if msg.expect_service_type.lower() not in svc:
            failures.append(
                f"service_type='{result.get('service_type')}' "
                f"(expected to contain '{msg.expect_service_type}')"
            )

    if msg.expect_doors_count is not None:
        got = result.get("doors_count")
        if got != msg.expect_doors_count:
            failures.append(
                f"doors_count={got} (expected {msg.expect_doors_count})"
            )

    if msg.expect_no_price:
        reply_lower = reply.lower()
        found = [p for p in _PRICE_PATTERNS if p in reply_lower]
        if found:
            failures.append(f"reply contains price hint: {found}")

    if msg.expect_text_contains:
        if msg.expect_text_contains.lower() not in reply.lower():
            failures.append(
                f"reply missing '{msg.expect_text_contains}'"
            )

    if msg.expect_text_not_contains:
        if msg.expect_text_not_contains.lower() in reply.lower():
            failures.append(
                f"reply should NOT contain '{msg.expect_text_not_contains}'"
            )

    return failures


async def run_scenario(
    scenario: Scenario,
    get_reply_fn,
    api_key: str,
    clear_fn,
) -> ScenarioResult:
    sender = f"autotest_{scenario.id}@c.us"
    clear_fn(sender)

    result = ScenarioResult(scenario=scenario)
    t0 = time.monotonic()

    try:
        for i, msg in enumerate(scenario.messages):
            resp = await get_reply_fn(sender, msg.text, api_key, mock_claude=False)
            reply = resp.get("reply_text", "")
            failures = _check_message(msg, resp, reply)
            step = StepResult(
                turn=i + 1,
                message=msg.text,
                reply=reply,
                passed=len(failures) == 0,
                failures=failures,
                data={
                    "handoff": resp.get("handoff_to_human"),
                    "service_type": resp.get("service_type"),
                    "doors_count": resp.get("doors_count"),
                    "city": resp.get("city"),
                    "full_name": resp.get("full_name"),
                    "phone": resp.get("phone"),
                },
            )
            result.steps.append(step)
            if failures:
                result.passed = False
    except Exception as exc:
        result.passed = False
        result.error = str(exc)
    finally:
        clear_fn(sender)
        result.duration_ms = int((time.monotonic() - t0) * 1000)

    return result


async def run_all(get_reply_fn, api_key: str, clear_fn) -> list[ScenarioResult]:
    results = []
    for scenario in SCENARIOS:
        r = await run_scenario(scenario, get_reply_fn, api_key, clear_fn)
        results.append(r)
    return results
