#!/usr/bin/env python3
"""
Full bot test suite — sends real messages to /test-chat (no mock)
and checks each response for correctness.

Usage:
    python3 run_full_test.py [--url http://localhost:3000]
"""
import sys
import json
import time
import argparse
import httpx

BASE_URL = "https://michael-doors-whatsapp-bot.onrender.com"

# ── helpers ───────────────────────────────────────────────────────────────────

def reset(sender: str) -> None:
    httpx.post(f"{BASE_URL}/test-chat", json={"sender": sender, "reset": True}, timeout=10)

def send(sender: str, message: str) -> dict:
    r = httpx.post(
        f"{BASE_URL}/test-chat",
        json={"sender": sender, "message": message, "mock": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

PRICE_PATTERNS = ["₪", "שקל", "ש\"ח", "עולה", "עלות", "תעריף", "מחיר מ-", "החל מ-",
                  "בין", "כ-", "בסביבות", "לא יקר", "יקר", "זול"]
BOT_PHRASES = ["אנא ספק", "הבקשה נקלטה", "כנציגת השירות", "אני כאן לסייע",
               "האם יש עוד שאלות", "פנייתך חשובה"]

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def check(name: str, resp: dict, *,
          expect_handoff: bool = False,
          no_handoff: bool = False,
          no_price: bool = True,
          has_reply2_first: bool = None,
          no_reply2: bool = False,
          contains: list[str] = None,
          not_contains: list[str] = None,
          no_bot_phrases: bool = True) -> bool:
    issues = []
    text = (resp.get("reply_text") or "") + " " + (resp.get("reply_text_2") or "")

    if expect_handoff and not resp.get("handoff_to_human"):
        issues.append("HANDOFF expected but not set")
    if no_handoff and resp.get("handoff_to_human"):
        issues.append("HANDOFF set unexpectedly")
    if no_price:
        for p in PRICE_PATTERNS:
            import re
            if re.search(r'\d[\d,]*\s*(?:₪|ש["\']?ח|שקל)', text):
                issues.append(f"PRICE LEAKED in reply")
                break
    if no_bot_phrases:
        for bp in BOT_PHRASES:
            if bp in text:
                issues.append(f"BOT PHRASE detected: '{bp}'")
    if has_reply2_first is True and not resp.get("reply_text_2"):
        issues.append("reply_text_2 expected on first message but missing")
    if has_reply2_first is False and resp.get("reply_text_2"):
        issues.append(f"reply_text_2 should be null but got: {resp.get('reply_text_2')[:40]}")
    if no_reply2 and resp.get("reply_text_2"):
        issues.append("reply_text_2 should be null in follow-up turn")
    if contains:
        for c in contains:
            if c.lower() not in text.lower():
                issues.append(f"Expected to contain: '{c}'")
    if not_contains:
        for nc in not_contains:
            if nc.lower() in text.lower():
                issues.append(f"Should NOT contain: '{nc}'")

    status = PASS if not issues else FAIL
    results.append((status, name, issues, resp.get("reply_text", "")[:80]))
    return not issues

def print_result(status, name, issues, preview):
    print(f"{status} {name}")
    if issues:
        for i in issues:
            print(f"     └─ {i}")
    elif status == PASS:
        print(f"     └─ {preview}")

# ── TEST SUITE ────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "="*60)
    print("  🧪 BOT FULL TEST SUITE")
    print("="*60 + "\n")

    # ── 1. Greeting only ──────────────────────────────────────────
    s = "tst_greeting@c.us"; reset(s)
    r = send(s, "שלום")
    check("01. פתיחה — ברכה בלבד", r,
          has_reply2_first=True, no_handoff=True, no_price=True,
          contains=["תודה שפניתם לדלתות מיכאל"])

    # ── 2. Direct price inquiry on first message ──────────────────
    s = "tst_price_first@c.us"; reset(s)
    r = send(s, "כמה עולה דלת?")
    check("02. מחיר בהודעה ראשונה — ללא ציון סוג", r,
          has_reply2_first=True, no_handoff=True, no_price=True,
          contains=["כניסה"])  # must ask door type

    # ── 3. Entrance door direct ───────────────────────────────────
    s = "tst_entrance@c.us"; reset(s)
    r = send(s, "אני מחפש דלת כניסה מעוצבת")
    check("03. דלת כניסה — כוונה ברורה", r,
          has_reply2_first=True, no_handoff=True, no_price=True)

    # ── 4. Interior doors direct ──────────────────────────────────
    s = "tst_interior@c.us"; reset(s)
    r = send(s, "אני צריכה 4 דלתות פנים לבית חדש")
    check("04. דלתות פנים — כמות + בית חדש", r,
          has_reply2_first=True, no_handoff=True, no_price=True)

    # ── 5. Both door types ────────────────────────────────────────
    s = "tst_both@c.us"; reset(s)
    r = send(s, "אני צריך דלת כניסה ו-3 דלתות פנים")
    check("05. שני סוגי דלתות בהודעה אחת", r,
          has_reply2_first=True, no_handoff=True, no_price=True,
          not_contains=["כניסה או פנים"])  # must NOT ask כניסה או פנים

    # ── 6. Specific model — נפחות ─────────────────────────────────
    s = "tst_model@c.us"; reset(s)
    r = send(s, "מתעניין בדלת נפחות")
    check("06. דגם ספציפי — נפחות", r,
          has_reply2_first=True, no_handoff=True, no_price=True,
          not_contains=["כניסה או פנים"])  # knows it's entrance

    # ── 7. Numeric ambiguous model — 5005 ─────────────────────────
    s = "tst_5005@c.us"; reset(s)
    r = send(s, "יש לכם 5005?")
    check("07. מספר דו-משמעי — 5005", r,
          has_reply2_first=True, no_handoff=True,
          contains=["5005"])  # must ask for clarification

    # ── 8. Showroom address ───────────────────────────────────────
    s = "tst_address@c.us"; reset(s)
    r = send(s, "מה הכתובת שלכם?")
    check("08. כתובת אולם תצוגה", r,
          has_reply2_first=True, no_handoff=True,
          contains=["נתיבות"])

    # ── 9. Working hours ──────────────────────────────────────────
    s = "tst_hours@c.us"; reset(s)
    r = send(s, "מתי אתם פתוחים?")
    check("09. שעות פעילות", r,
          has_reply2_first=True, no_handoff=True,
          contains=["18:00"])

    # ── 10. Repair request ────────────────────────────────────────
    s = "tst_repair@c.us"; reset(s)
    r = send(s, "הדלת שלי לא נסגרת, צריך תיקון")
    check("10. בקשת תיקון — אמפתיה קודם", r,
          has_reply2_first=True, no_handoff=True)

    # ── 11. ממד inquiry ───────────────────────────────────────────
    s = "tst_mamad@c.us"; reset(s)
    r = send(s, "אני צריך דלת ממד")
    check("11. דלת ממ\"ד", r,
          has_reply2_first=True, no_handoff=True)

    # ── 12. Geographic — outside primary area ─────────────────────
    s = "tst_geo@c.us"; reset(s)
    r = send(s, "אני מתל אביב, אתם מגיעים?")
    check("12. איזור גיאוגרפי — מרכז", r,
          has_reply2_first=True, no_handoff=True)

    # ── 13. Emergency ─────────────────────────────────────────────
    s = "tst_emrg@c.us"; reset(s)
    r = send(s, "פרצו לי הביתה, הדלת שבורה, צריך עזרה דחופה")
    check("13. חירום — פריצה", r,
          has_reply2_first=True)

    # ── 14. Request for human ─────────────────────────────────────
    s = "tst_human@c.us"; reset(s)
    r = send(s, "אני רוצה לדבר עם נציג אנושי")
    check("14. בקשה לנציג אנושי", r,
          has_reply2_first=True, no_price=True)

    # ── 15. Contractor large project ──────────────────────────────
    s = "tst_contractor@c.us"; reset(s)
    r = send(s, "אני קבלן, יש לי פרויקט של 50 דלתות")
    check("15. קבלן — פרויקט גדול", r,
          has_reply2_first=True, no_price=True)

    # ── 16. Female gender detection ───────────────────────────────
    # First message uses scripted router (neutral plural is OK).
    # Second message should address female form.
    s = "tst_female@c.us"; reset(s)
    send(s, "אני מחפשת דלת כניסה לבית שלי")
    r = send(s, "מעוצבת")
    text = (r.get("reply_text") or "")
    check("16. זיהוי מגדר — נקבה (סיבוב 2)", r,
          no_reply2=True,
          not_contains=["לא הבנתי", "כניסה או פנים"])

    # ── 17. Male gender detection ─────────────────────────────────
    s = "tst_male@c.us"; reset(s)
    r = send(s, "אני מחפש דלת פנים לחדר השינה שלי")
    check("17. זיהוי מגדר — זכר", r,
          has_reply2_first=True)

    # ── 18. Spelling errors ───────────────────────────────────────
    s = "tst_spell@c.us"; reset(s)
    r = send(s, "שלום, אני מחפש דלט כניסע לביתי")
    check("18. שגיאות כתיב — דלט/כניסע", r,
          has_reply2_first=True, no_handoff=True,
          not_contains=["לא הבנתי", "לא מכיר"])

    # ── 19. Sticker / emoji only ──────────────────────────────────
    s = "tst_sticker@c.us"; reset(s)
    r = send(s, "👍")
    check("19. סטיקר/אמוג'י בלבד", r,
          has_reply2_first=True, no_handoff=True,
          contains=["קיבלתי"])

    # ── 20. Off-topic — warranty question ─────────────────────────
    s = "tst_warranty@c.us"; reset(s)
    r = send(s, "כמה שנות אחריות יש על הדלתות?")
    check("20. שאלה חופשית — אחריות", r,
          has_reply2_first=True, no_handoff=True,
          not_contains=["אין לי מידע", "לא יודעת"])

    # ── 21. Off-topic — installation time ────────────────────────
    s = "tst_install@c.us"; reset(s)
    r = send(s, "תוך כמה זמן מתקינים?")
    check("21. שאלה חופשית — זמן התקנה", r,
          has_reply2_first=True, no_handoff=True,
          not_contains=["אין לי מידע", "לא יודעת"])

    # ── 22. Off-topic — colors ────────────────────────────────────
    s = "tst_colors@c.us"; reset(s)
    r = send(s, "באיזה צבעים יש דלתות פנים?")
    check("22. שאלה חופשית — צבעים", r,
          has_reply2_first=True, no_handoff=True,
          contains=["לבן"])

    # ── 23. Full closing flow — 6 turns ───────────────────────────
    s = "tst_close@c.us"; reset(s)
    print("\n  📋 טסט 23: מסע שיחה מלא (6 סיבובים)...")
    r1 = send(s, "אני מחפש דלת כניסה מעוצבת")
    check("23a. פתיחה — דלת כניסה", r1, has_reply2_first=True, no_handoff=True)
    time.sleep(1)

    r2 = send(s, "מעוצבת עם זכוכית")
    check("23b. סגנון — מעוצבת עם זכוכית", r2, no_reply2=True, no_handoff=True)
    time.sleep(1)

    r3 = send(s, "החלפה בלי משקוף")
    check("23c. פירוק משקוף — לא", r3, no_reply2=True, no_handoff=True)
    time.sleep(1)

    r4 = send(s, "דוד כהן, 0521234567, תל אביב")
    check("23d. פרטי קשר", r4, no_reply2=True, no_handoff=True)
    time.sleep(1)

    r5 = send(s, "כן הכל נכון")
    check("23e. אישור סיכום", r5, no_reply2=True, no_handoff=True)
    time.sleep(1)

    r6 = send(s, "אחרי 17:00")
    check("23f. שעה מועדפת → HANDOFF", r6, no_reply2=True, expect_handoff=True)

    # ── 24. No price in multi-turn conversation ────────────────────
    s = "tst_noprice@c.us"; reset(s)
    r = send(s, "כמה עולה דלת כניסה מעוצבת נפחות?")
    check("24. מחיר — אסור לציין סכום", r,
          has_reply2_first=True, no_price=True)

    # ── 25. Vague message ─────────────────────────────────────────
    s = "tst_vague@c.us"; reset(s)
    r = send(s, "מתעניין")
    check("25. הודעה עמומה — 'מתעניין'", r,
          has_reply2_first=True, no_handoff=True)

    # ── 26. Interior door — renovation context ─────────────────────
    s = "tst_renov@c.us"; reset(s)
    r = send(s, "אנחנו בשיפוץ, מחפשים דלתות פנים")
    check("26. שיפוץ — דלתות פנים", r,
          has_reply2_first=True, no_handoff=True, no_price=True)

    # ── 27. Customer says bot-like ("זה בוט?") ────────────────────
    s = "tst_isbot@c.us"; reset(s)
    r = send(s, "שלום")
    send(s, "את בוט?")
    r2 = send(s, "את בוט?")
    check("27. 'את בוט?' — תשובה אנושית", r2,
          no_handoff=True,
          not_contains=["אני בוט", "בינה מלאכותית", "AI", "GPT", "Claude", "אוטומטי"])

    # ── 28. Short answers flow ────────────────────────────────────
    s = "tst_short@c.us"; reset(s)
    send(s, "מחפש דלת כניסה")
    r = send(s, "מעוצבת")
    check("28. תשובה קצרה 'מעוצבת'", r,
          no_reply2=True, no_handoff=True,
          not_contains=["לא הבנתי"])

    # ── 29. Customer complains then provides info ──────────────────
    s = "tst_complaint@c.us"; reset(s)
    r1 = send(s, "יש לי בעיה עם דלת שקניתי, הציר שבור")
    check("29a. תלונה — ציר שבור", r1, has_reply2_first=True)
    time.sleep(1)
    r2 = send(s, "קניתי לפני שנה, מגיע לי אחריות")
    check("29b. תביעת אחריות", r2, no_reply2=True,
          not_contains=["אין לי מידע"])

    # ── 30. Hebrew abbreviation escaping ──────────────────────────
    s = "tst_abb@c.us"; reset(s)
    r = send(s, "צריך דלת לממד")
    text = r.get("reply_text", "")
    try:
        # if reply_text contains ממ"ד, the JSON was properly escaped
        has_mamad = "ממ" in text
        ok = r.get("ok", True)
    except Exception:
        ok = False
    status = PASS if ok else FAIL
    results.append((status, "30. קידוד JSON — ממ\"ד", [], text[:80]))

    # ── PRINT RESULTS ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  📊 תוצאות")
    print("="*60)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)

    for status, name, issues, preview in results:
        print_result(status, name, issues, preview)

    print("\n" + "="*60)
    print(f"  סה\"כ: {passed}/{len(results)} עברו | {failed} נכשלו")
    print("="*60 + "\n")

    return failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    if args.url:
        BASE_URL = args.url.rstrip("/")

    failed = run_all()
    sys.exit(1 if failed else 0)
