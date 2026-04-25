"""
messages.py — Fixed messages only (no flow / decision logic).
All flow decisions live in simple_router._decide_next_action().

Contents:
  PITCH               — company introduction (first message only)
  CONTACT_OPENER      — Stage 4 opener (sent alone, no question appended)
  STAGE3_QUESTION     — pre-contact wrap-up (sent once per conversation)
  FINAL_HANDOFF       — Stage 7 farewell

  QUESTION_TEMPLATES  — keyed by template_key; Claude uses these for phrasing.
  ERROR_MSG           — system error replies
"""

# ── Fixed messages ────────────────────────────────────────────────────────────

PITCH: str = (
    "תודה שפניתם למיכאל דלתות 🚪\n"
    "אנו מציעים דלתות כניסה ופנים בסטנדרט הגבוה ביותר — התקנה, מכירה ואחריות מעל שנתיים ✨\n"
    "באולם התצוגה שלנו בנתיבות תוכלו להתרשם ממגוון רחב של דגמים 😊 (מומלץ לתאם פגישה מראש)"
)

CONTACT_OPENER: str = (
    "כדי שנציג יוכל לחזור אליכם עם התאמה מסודרת, אשמח לפרטים ליצירת קשר."
)

STAGE3_QUESTION: str = "יש עוד משהו ספציפי שחשוב לכם?"

FINAL_HANDOFF: str = "מעולה! נחזור אליכם בהקדם עם הצעת מחיר מסודרת 😊 יום נפלא! 💙"

# ── Phrasing templates ────────────────────────────────────────────────────────
# Keyed by template_key from NextAction.
# Claude receives the template_key and uses the matching phrasing.
# Claude adapts the wording for gender (see system prompt) but never changes the content.

QUESTION_TEMPLATES: dict[str, str] = {
    # Entrance doors queue
    "ask_entrance_scope":        "האם מדובר בדלת כולל משקוף, או דלת בלבד?",
    "ask_entrance_style":        "דלת חלקה או מעוצבת?",

    # Interior doors queue
    "ask_interior_project_type": "מדובר בבית חדש, שיפוץ, או החלפה של דלתות קיימות?",
    "ask_interior_quantity":     "כמה דלתות פנים בערך?",
    "ask_interior_style":        "הדלתות חלקות או מעוצבות?",

    # Mamad queue
    "ask_mamad_type":            'ממ"ד חדש, או יש ממ"ד קיים שרוצים להחליף את דלתו?',

    # Repair queue
    "ask_repair_type":           "מדובר בדלת כניסה או דלת פנים?",

    # Contact collection
    "ask_phone":                 "מה מספר הטלפון?",
    "ask_name":                  "על שם מי הפנייה?",
    "ask_city":                  "באיזו עיר מדובר?",

    # Callback time (gender-specific variants)
    "ask_callback_time_neutral": "מתי נוח שנחזור אליכם?",
    "ask_callback_time_female":  "מתי נוח שנחזור אלייך?",
    "ask_callback_time_male":    "מתי נוח שנחזור אליך?",

    # Fallbacks
    "ask_topic_clarification":   'לגבי איזה סוג דלת מדובר — כניסה, פנים, או ממ"ד?',
    "ask_safe_fallback":         "תוכלו לספר לי קצת יותר על מה שאתם צריכים?",

    # Showroom
    "showroom_address":          "בעלי המלאכה 15, נתיבות 📍 מומלץ לתאם פגישה מראש — רוצים שנקבע?",

    # Catalogs (fixed messages — sent verbatim)
    "entrance_catalog":          "הנה הקטלוג לדלתות כניסה מעוצבות: https://www.michaeldoors.co.il/catalog/entry-designed 😊",
    "interior_catalog":          "הנה הקטלוג לדלתות פנים: https://www.michaeldoors.co.il/catalog/interior-smooth 😊",

    # Model/style follow-up (after catalog is sent)
    "ask_entrance_model":        "יש דגם ספציפי שתפס אתכם ושתרצו שנכלול בהצעה?",
    "ask_interior_model":        "יש סגנון ספציפי שתפס אתכם ושתרצו שנכלול בהצעה?",

    # Mamad scope (same wording as entrance scope but different field)
    "ask_mamad_scope":           "האם מדובר בדלת כולל משקוף, או דלת בלבד?",

    # Mamad technical info (fixed — sent after mamad scope is collected)
    "mamad_info":                'דלת ממ"ד צריכה לעמוד בתקן ת"י 5044 — אנחנו מסדרים הכל כולל אישור הג"א. המחיר מתברר אחרי מדידה 😊',

    # Repair intro (fixed)
    "repair_intro":              "אוי, זה לא נעים 💙 ספרו לי מה קורה — ננסה לטפל בזה.",

    # Showroom contact opener variant
    "contact_opener_showroom":   "כן, אפשר להגיע לאולם התצוגה. כדי שנציג יתאם איתכם אישית, אשמח לפרטים ליצירת קשר.",
}

# ── Error messages ────────────────────────────────────────────────────────────
ERROR_MSG: dict[str, str] = {
    "api_error":   "רגע, בודקת 😊 תכתבו לי שוב בעוד רגע ואענה לכם",
    "parse_error": "רגע, בודקת 😊 תכתבו לי שוב בעוד רגע ואענה לכם",
}
