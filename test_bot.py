#!/usr/bin/env python3
"""
Test script — calls get_reply directly, no Green API needed.
Run: python test_bot.py
Requires ANTHROPIC_API_KEY in .env (or environment).
"""
import asyncio
import os
import sys
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not API_KEY:
    print("[FATAL] ANTHROPIC_API_KEY not set in .env")
    sys.exit(1)

# Patch environment so simple_router can import config without crashing
os.environ.setdefault("GREEN_API_INSTANCE_ID", "test")
os.environ.setdefault("GREEN_API_TOKEN", "test")
os.environ.setdefault("TEST_MODE", "true")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from michael_doors_bot.engine.simple_router import get_reply, _conversations

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    results.append((label, condition))


async def run_turn(sender: str, msg: str) -> dict:
    print(f"\n  >> {msg!r}")
    result = await get_reply(sender, msg, API_KEY)
    print(f"     pulse1: {result['reply_text'][:80]!r}")
    if result.get("reply_text_2"):
        print(f"     pulse2: {result['reply_text_2'][:80]!r}")
    return result


# ─── Scenario 1: First message triggers scripted response (interior door) ─────

async def test_scripted_first_message():
    print("\n=== TEST 1: Scripted first message — interior door ===")
    sender = "TEST_INTERIOR_01@c.us"
    _conversations.pop(sender, None)

    r = await run_turn(sender, "אני מחפש דלתות פנים לדירה")

    check("Has reply_text",   bool(r.get("reply_text")))
    check("reply_text_2 is null", r.get("reply_text_2") is None)
    check("Has greeting",
          "תודה שפניתם" in r["reply_text"],
          r["reply_text"][:60])
    check("Has actual response (not error)",
          "בודקת" not in r["reply_text"],
          r["reply_text"][:80])
    check("Not handoff",      not r.get("handoff_to_human"))


# ─── Scenario 2: THE BUG — second message after scripted first must get Claude ─

async def test_second_message_after_scripted():
    print("\n=== TEST 2: Second message after scripted response (THE BUG) ===")
    sender = "TEST_INTERIOR_02@c.us"
    _conversations.pop(sender, None)

    await run_turn(sender, "מחפש דלתות פנים")
    r = await run_turn(sender, "14 יחידות פולימר מלא")

    check("Got reply_text",     bool(r.get("reply_text")))
    check("Not an error reply", "בודקת" not in r.get("reply_text", ""),
          r.get("reply_text", "")[:60])
    check("reply_text_2 is null on turn 2",
          r.get("reply_text_2") is None,
          f"got: {r.get('reply_text_2')!r}")
    check("Relevant response (has דלת/יחידות/פולימר/כמה/מה/?)",
          any(w in r.get("reply_text", "") for w in ["דלת", "פולימר", "יחידות", "כמה", "מה", "?"]),
          r.get("reply_text", "")[:80])


# ─── Scenario 3: Greeting only ─────────────────────────────────────────────────

async def test_greeting():
    print("\n=== TEST 3: Greeting-only first message ===")
    sender = "TEST_GREET_03@c.us"
    _conversations.pop(sender, None)

    r = await run_turn(sender, "שלום")

    check("Has reply_text", bool(r.get("reply_text")))
    check("reply_text_2 is null", r.get("reply_text_2") is None)
    check("Asks how to help",
          any(w in r.get("reply_text", "") for w in ["במה", "לעזור", "?"]),
          r.get("reply_text", "")[:60])


# ─── Scenario 4: Price inquiry ─────────────────────────────────────────────────

async def test_price_inquiry():
    print("\n=== TEST 4: Price inquiry first message ===")
    sender = "TEST_PRICE_04@c.us"
    _conversations.pop(sender, None)

    r = await run_turn(sender, "כמה עולה דלת כניסה?")

    check("Has reply_text", bool(r.get("reply_text")))
    check("reply_text_2 is null", r.get("reply_text_2") is None)
    check("No explicit price leaked (scrubbed)",
          "₪" not in r.get("reply_text", "") and "₪" not in (r.get("reply_text_2") or ""))


# ─── Scenario 5: Multi-turn conversation ───────────────────────────────────────

async def test_multi_turn():
    print("\n=== TEST 5: Multi-turn — repair inquiry ===")
    sender = "TEST_MULTI_05@c.us"
    _conversations.pop(sender, None)

    r1 = await run_turn(sender, "יש לי בעיה בדלת הכניסה")
    check("Turn1 has greeting+response", "תודה שפניתם" in r1.get("reply_text", "") and "בודקת" not in r1.get("reply_text", ""))

    r2 = await run_turn(sender, "הדלת לא נסגרת עד הסוף")
    check("Turn2 not error",  "בודקת" not in r2.get("reply_text", ""),
          r2.get("reply_text", "")[:60])
    check("Turn2 pulse2 null", r2.get("reply_text_2") is None)

    r3 = await run_turn(sender, "קניתי את הדלת מכם לפני שנה")
    check("Turn3 relevant", bool(r3.get("reply_text")),
          r3.get("reply_text", "")[:60])


# ─── Scenario 6: Handoff detection ─────────────────────────────────────────────

async def test_handoff():
    print("\n=== TEST 6: Handoff to human ===")
    sender = "TEST_HANDOFF_06@c.us"
    _conversations.pop(sender, None)

    # Build context first
    await run_turn(sender, "אני רוצה להזמין דלת כניסה")
    await run_turn(sender, "פנורמי")
    await run_turn(sender, "השם שלי יוסי כהן")
    await run_turn(sender, "הטלפון שלי 0521234567")
    r = await run_turn(sender, "נתיבות")

    # After collecting all details, Claude should either handoff or give summary
    check("Got reply", bool(r.get("reply_text")),
          r.get("reply_text", "")[:60])


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Michael Doors Bot — Integration Tests (no Green API)")
    print("=" * 60)

    await test_scripted_first_message()
    await test_second_message_after_scripted()
    await test_greeting()
    await test_price_inquiry()
    await test_multi_turn()
    await test_handoff()

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("\033[32mAll tests passed — safe to reconnect WhatsApp\033[0m")
    else:
        failed = [label for label, ok in results if not ok]
        print(f"\033[31mFailed:\033[0m {', '.join(failed)}")
    print("=" * 60)
    return passed == total


# ─── הבדיקות שלי ──────────────────────────────────────────────────────────────
# כתבי כאן הודעות שבאת לבדוק, כמו שיחה אמיתית עם לקוח.
# כל run_turn = הודעה אחת שהלקוח שולח.
# אחרי כל שינוי בתסריט — הריצי: .venv/bin/python3 test_bot.py

MY_MESSAGES = [
    # שנה את ההודעות האלו לכל מה שתרצי לבדוק:
    "שלום, אני מחפש דלת כניסה לדירה חדשה",
    "אני גר בנתיבות",
    "מה יש לכם במבצע?",
]

async def test_my_conversation():
    if not MY_MESSAGES:
        return
    print("\n" + "=" * 60)
    print("הבדיקה שלי")
    print("=" * 60)
    sender = "MY_CUSTOM_TEST@c.us"
    _conversations.pop(sender, None)
    for msg in MY_MESSAGES:
        await run_turn(sender, msg)


if __name__ == "__main__":
    async def run_all():
        await main()
        await test_my_conversation()
    asyncio.run(run_all())
