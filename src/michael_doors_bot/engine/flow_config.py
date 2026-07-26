"""
flow_config.py — Declarative configuration for the DETERMINISTIC bot engine.

This module contains ONLY data: step names, the exact fixed messages, the
option→value tables (with explicit aliases), and human-readable labels.
There is NO logic here and NO external dependency — it is pure data so it can
be imported and tested in isolation.

The deterministic engine (flow_machine.py) reads these tables. It NEVER calls
an AI model, never guesses intent, and never does broad fuzzy matching.

Business: דלתות מיכאל (Michael Doors). The bot is an automated system — it has
no name and no human persona.
"""

BUSINESS_NAME = "דלתות מיכאל"

# ── Step names (State Machine states) ─────────────────────────────────────────
START                 = "START"                  # nothing sent yet
ASK_INQUIRY_TYPE      = "ASK_INQUIRY_TYPE"
ASK_ENTRANCE_FRAME    = "ASK_ENTRANCE_FRAME"
ASK_INTERIOR_QUANTITY = "ASK_INTERIOR_QUANTITY"
ASK_OTHER_DETAILS     = "ASK_OTHER_DETAILS"
ASK_PROJECT_TYPE      = "ASK_PROJECT_TYPE"
ASK_CONTACT_DETAILS   = "ASK_CONTACT_DETAILS"
ASK_MISSING_NAME      = "ASK_MISSING_NAME"
ASK_MISSING_CITY      = "ASK_MISSING_CITY"
ASK_MISSING_PHONE     = "ASK_MISSING_PHONE"
CONFIRM_DETAILS       = "CONFIRM_DETAILS"
WAITING_FOR_AGENT     = "WAITING_FOR_AGENT"       # terminal — bot silent

# ── conversationMode values ───────────────────────────────────────────────────
MODE_BOT_COLLECTING    = "BOT_COLLECTING"
MODE_WAITING_FOR_AGENT = "WAITING_FOR_AGENT"
MODE_HUMAN_ACTIVE      = "HUMAN_ACTIVE"

# ── handoff reasons ───────────────────────────────────────────────────────────
HANDOFF_INVALID_RESPONSES   = "INVALID_RESPONSES"
HANDOFF_CUSTOMER_REQUESTED  = "CUSTOMER_REQUESTED_AGENT"
HANDOFF_READY               = "READY_FOR_AGENT"   # normal completion (not a failure)

# ── inquiryStatus values ──────────────────────────────────────────────────────
STATUS_COLLECTING = "בתהליך איסוף פרטים"
STATUS_READY      = "ממתין לנציג"

MAX_INVALID_ATTEMPTS = 2  # after 2 invalid answers in one step → handoff

# ══════════════════════════════════════════════════════════════════════════════
# FIXED MESSAGES (verbatim from the approved spec)
# ══════════════════════════════════════════════════════════════════════════════

MSG_OPENING = (
    "שלום 👋\n"
    f"תודה שפנית ל-{BUSINESS_NAME}.\n\n"
    "זהו מענה אוטומטי קצר שנועד לאסוף את פרטי הפנייה ולהעביר אותם לנציג שלנו.\n\n"
    "בכל שלב נא להשיב רק לפי האפשרויות שמופיעות בהודעה.\n\n"
    "מה נושא הפנייה?\n\n"
    "1. דלת כניסה\n"
    "2. דלתות פנים\n"
    "3. דלת ממ\"ד\n"
    "4. אחר\n\n"
    "נא להשיב במספר האפשרות בלבד."
)

MSG_ENTRANCE_FRAME = (
    "האם מדובר בדלת כניסה עם משקוף או ללא משקוף?\n\n"
    "1. עם משקוף\n"
    "2. ללא משקוף\n"
    "3. יש משקוף קיים\n"
    "4. לא בטוח/ה\n\n"
    "נא להשיב במספר האפשרות בלבד."
)

MSG_INTERIOR_QUANTITY = (
    "בכמה דלתות פנים מדובר?\n\n"
    "1. דלת אחת\n"
    "2. 2–3 דלתות\n"
    "3. 4–6 דלתות\n"
    "4. 7 דלתות ומעלה\n"
    "5. עדיין לא ידוע\n\n"
    "נא להשיב במספר האפשרות בלבד."
)

MSG_OTHER_DETAILS = "אפשר לכתוב במשפט קצר במה נוכל לעזור?"

MSG_PROJECT_TYPE = (
    "באיזה סוג פרויקט מדובר?\n\n"
    "1. שיפוץ\n"
    "2. בית חדש\n"
    "3. החלפת דלת קיימת\n"
    "4. אחר או עדיין לא ידוע\n\n"
    "נא להשיב במספר האפשרות בלבד."
)

MSG_CONTACT_DETAILS = (
    "מעולה, נשארו רק פרטי יצירת קשר.\n\n"
    "נא לשלוח בהודעה אחת:\n\n"
    "שם מלא:\n"
    "עיר:\n"
    "מספר טלפון:\n\n"
    "לדוגמה:\n\n"
    "שם מלא: ישראל ישראלי\n"
    "עיר: נתיבות\n"
    "מספר טלפון: 0501234567"
)

MSG_MISSING_NAME  = "תודה. חסר לנו רק השם המלא.\nנא לשלוח שם מלא."
MSG_MISSING_CITY  = "תודה. חסרה לנו רק העיר או היישוב.\nנא לשלוח את שם העיר."
MSG_MISSING_PHONE = "תודה. חסר לנו רק מספר טלפון ליצירת קשר.\nנא לשלוח מספר טלפון."

MSG_READY = (
    "תודה ✅\n\n"
    "הפרטים התקבלו והועברו לנציג שלנו.\n\n"
    "מכאן נציג אנושי ימשיך איתך את הטיפול באותה שיחת WhatsApp."
)

MSG_AGENT_HANDOFF = (
    "כמובן. העברנו את השיחה לנציג אנושי והוא ימשיך איתך כאן בהקדם."
)

MSG_INVALID_HANDOFF = (
    "לא הצלחנו להשלים את הפרטים באופן אוטומטי.\n"
    "העברנו את השיחה לנציג אנושי שימשיך איתך כאן."
)

MSG_PRICE = (
    "המחיר משתנה בהתאם לסוג הדלת ולפרטי הפרויקט. "
    "נציג שלנו יוכל לתת מענה מדויק לאחר קבלת הפרטים."
)

# Prefix for "we saved your question" — the current step's question is appended.
MSG_SAVED_QUESTION_PREFIX = (
    "שמרנו את השאלה שלך כדי שהנציג יוכל להתייחס אליה.\n\n"
    "כדי להשלים את הפנייה:\n"
)

# Media received during collection — acknowledge, then re-ask current step.
MSG_MEDIA_PREFIX = (
    "קיבלנו את הקובץ ששלחת ונעביר אותו לנציג.\n\n"
    "כדי להשלים את פרטי הפנייה:\n"
)

# ══════════════════════════════════════════════════════════════════════════════
# OPTION TABLES  (raw alias → canonical value)
# Keys are normalized by flow_machine before matching, so write them naturally.
# NO fuzzy matching — only these exact (normalized) strings are accepted.
# ══════════════════════════════════════════════════════════════════════════════

OPTIONS_INQUIRY_TYPE = {
    "1": "ENTRANCE_DOOR", "כניסה": "ENTRANCE_DOOR", "דלת כניסה": "ENTRANCE_DOOR",
    "2": "INTERIOR_DOORS", "פנים": "INTERIOR_DOORS",
    "דלת פנים": "INTERIOR_DOORS", "דלתות פנים": "INTERIOR_DOORS",
    "פולימר": "INTERIOR_DOORS", "דלת פולימר": "INTERIOR_DOORS",
    "דלתות פולימר": "INTERIOR_DOORS", "דלתות פנים פולימר": "INTERIOR_DOORS",
    "3": "MAMAD_DOOR", "ממד": "MAMAD_DOOR", "ממ\"ד": "MAMAD_DOOR",
    "דלת ממד": "MAMAD_DOOR", "דלת ממ\"ד": "MAMAD_DOOR",
    "4": "OTHER", "אחר": "OTHER",
}

OPTIONS_ENTRANCE_FRAME = {
    "1": "WITH_FRAME", "עם משקוף": "WITH_FRAME",
    "2": "WITHOUT_FRAME", "ללא משקוף": "WITHOUT_FRAME", "בלי משקוף": "WITHOUT_FRAME",
    "3": "EXISTING_FRAME", "יש משקוף": "EXISTING_FRAME", "יש משקוף קיים": "EXISTING_FRAME",
    "4": "UNKNOWN_FRAME", "לא בטוח": "UNKNOWN_FRAME", "לא בטוחה": "UNKNOWN_FRAME",
    "לא יודע": "UNKNOWN_FRAME", "לא יודעת": "UNKNOWN_FRAME",
}

OPTIONS_INTERIOR_QUANTITY = {
    "1": "ONE", "דלת אחת": "ONE", "אחת": "ONE",
    "2": "TWO_TO_THREE",
    "3": "FOUR_TO_SIX",
    "4": "SEVEN_PLUS",
    "5": "UNKNOWN_QUANTITY", "לא ידוע": "UNKNOWN_QUANTITY", "עדיין לא ידוע": "UNKNOWN_QUANTITY",
}

OPTIONS_PROJECT_TYPE = {
    "1": "RENOVATION", "שיפוץ": "RENOVATION",
    "2": "NEW_HOME", "בית חדש": "NEW_HOME",
    "3": "REPLACEMENT", "החלפה": "REPLACEMENT", "החלפת דלת קיימת": "REPLACEMENT",
    "4": "OTHER_OR_UNKNOWN", "אחר": "OTHER_OR_UNKNOWN", "לא ידוע": "OTHER_OR_UNKNOWN",
}

OPTIONS_CONFIRM = {
    "1": "CONFIRM", "כן": "CONFIRM",
    "2": "RESTART",
    "3": "AGENT", "נציג": "AGENT",
}

# Returning-customer menu (used in a later phase; defined here for completeness).
OPTIONS_RETURNING = {
    "1": "CONTINUE",
    "2": "NEW",
    "3": "AGENT", "נציג": "AGENT",
}

# ── Global intent aliases (checked before normal step processing) ─────────────
# Agent request — matched as a substring of the normalized text (fixed allowlist).
AGENT_ALIASES = [
    "נציג", "בן אדם", "מענה אנושי", "שירות לקוחות",
    "רוצה לדבר עם מישהו", "אפשר לדבר עם נציג", "לדבר עם נציג",
    "שיחזרו אליי", "שיחזרו אלי", "לא רוצה בוט", "אדם אמיתי",
]

# Price question — matched as a substring (fixed allowlist).
PRICE_ALIASES = [
    "מחיר", "כמה עולה", "כמה זה עולה", "עלות", "מחירון", "כמה זה", "כמה יעלה",
]

# ══════════════════════════════════════════════════════════════════════════════
# LABELS  (canonical value → Hebrew label, for the summary / Airtable)
# ══════════════════════════════════════════════════════════════════════════════

LABELS_INQUIRY_TYPE = {
    "ENTRANCE_DOOR": "דלת כניסה",
    "INTERIOR_DOORS": "דלתות פנים",
    "MAMAD_DOOR": "דלת ממ\"ד",
    "OTHER": "אחר",
}

LABELS_FRAME = {
    "WITH_FRAME": "עם משקוף",
    "WITHOUT_FRAME": "ללא משקוף",
    "EXISTING_FRAME": "יש משקוף קיים",
    "UNKNOWN_FRAME": "לא בטוח/ה",
}

LABELS_QUANTITY = {
    "ONE": "דלת אחת",
    "TWO_TO_THREE": "2–3 דלתות",
    "FOUR_TO_SIX": "4–6 דלתות",
    "SEVEN_PLUS": "7 דלתות ומעלה",
    "UNKNOWN_QUANTITY": "עדיין לא ידוע",
}

LABELS_PROJECT = {
    "RENOVATION": "שיפוץ",
    "NEW_HOME": "בית חדש",
    "REPLACEMENT": "החלפת דלת קיימת",
    "OTHER_OR_UNKNOWN": "אחר או עדיין לא ידוע",
}

# ── Per-step lookup used by the engine ────────────────────────────────────────
# Maps a menu step → (its message, its option table). Free-text / custom steps
# (OTHER_DETAILS, CONTACT, MISSING_*, CONFIRM) are handled explicitly in the engine.
MENU_STEPS = {
    ASK_INQUIRY_TYPE:      (MSG_OPENING,            OPTIONS_INQUIRY_TYPE),
    ASK_ENTRANCE_FRAME:    (MSG_ENTRANCE_FRAME,     OPTIONS_ENTRANCE_FRAME),
    ASK_INTERIOR_QUANTITY: (MSG_INTERIOR_QUANTITY,  OPTIONS_INTERIOR_QUANTITY),
    ASK_PROJECT_TYPE:      (MSG_PROJECT_TYPE,       OPTIONS_PROJECT_TYPE),
}

# Prompt shown for a given step (used when we need to re-ask it).
STEP_PROMPT = {
    ASK_INQUIRY_TYPE:      MSG_OPENING,
    ASK_ENTRANCE_FRAME:    MSG_ENTRANCE_FRAME,
    ASK_INTERIOR_QUANTITY: MSG_INTERIOR_QUANTITY,
    ASK_OTHER_DETAILS:     MSG_OTHER_DETAILS,
    ASK_PROJECT_TYPE:      MSG_PROJECT_TYPE,
    ASK_CONTACT_DETAILS:   MSG_CONTACT_DETAILS,
    ASK_MISSING_NAME:      MSG_MISSING_NAME,
    ASK_MISSING_CITY:      MSG_MISSING_CITY,
    ASK_MISSING_PHONE:     MSG_MISSING_PHONE,
}
