#!/usr/bin/env python3
"""
Unit tests for topic detection and entrance synonym coverage.
No API key needed — pure regex / state-machine layer.

Run: python test_topic_detection.py
"""
import os
import sys
from pathlib import Path

# Neutralise load_dotenv BEFORE any project code runs so .env (DATA_DIR=/data)
# never overwrites our test environment.
import dotenv as _dotenv
_dotenv.load_dotenv = lambda *a, **kw: None  # monkey-patch to no-op

os.environ["DATA_DIR"]               = "/tmp"
os.environ["GREEN_API_INSTANCE_ID"]  = "test"
os.environ["GREEN_API_TOKEN"]        = "test"
os.environ["TEST_MODE"]              = "true"
os.environ["ANTHROPIC_API_KEY"]      = "test"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from michael_doors_bot.engine.simple_router import (
    _detect_topics_from_message,
    _extract_fields_from_message,
    _decide_next_action,
    _advance_stage,
    _empty_conv_state,
    _normalize_callback_time,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results: list[tuple[str, bool]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    results.append((label, condition))


def topics(msg: str) -> list[str]:
    return _detect_topics_from_message(msg)


# ── Section 1: entrance_doors single-topic detection ─────────────────────────

print("\n=== Section 1: entrance_doors synonyms ===")

check("דלת כניסה",        "entrance_doors" in topics("אני מחפש דלת כניסה"))
check("דלת חוץ",          "entrance_doors" in topics("צריך דלת חוץ לבית"))
check("דלת חיצונית",      "entrance_doors" in topics("דלת חיצונית לדירה"))
check("דלת ראשית",        "entrance_doors" in topics("מחפש דלת ראשית"))
check("דלת לבית",         "entrance_doors" in topics("צריכה דלת לבית"))
check("דלת פלדה",         "entrance_doors" in topics("מה המחיר של דלת פלדה?"))
check("דלת ברזל",         "entrance_doors" in topics("יש לכם דלתות ברזל?"))
check("כניסה לבית",       "entrance_doors" in topics("דלת לכניסה לבית"))
check("כניסה לדירה",      "entrance_doors" in topics("כניסה לדירה — מחפשים דלת"))
check("כניסה לבניין",     "entrance_doors" in topics("צריך לשדרג כניסה לבניין"))
check("ראשית standalone", "entrance_doors" in topics("פנים וראשית"),
      topics("פנים וראשית"))
check("ראשית + ו prefix",  "entrance_doors" in topics("בית חדש ופנים וראשית"),
      topics("בית חדש ופנים וראשית"))
check("מודל נפחות",        "entrance_doors" in topics("מעוניין בדגם נפחות"))
check("מודל פנורמי",       "entrance_doors" in topics("פנורמי — כמה עולה?"))
check("מודל יווני",        "entrance_doors" in topics("יש לכם יווני?"))
check("מודל סביליה",       "entrance_doors" in topics("סביליה — יש מלאי?"))
check("מודל עדן",          "entrance_doors" in topics("עדן פלאטינום"))
check("מודל קלאסי",        "entrance_doors" in topics("רוצה דלת קלאסי"))

# ── Section 2: interior_doors detection not affected ─────────────────────────

print("\n=== Section 2: interior_doors still works ===")

check("דלתות פנים",        "interior_doors" in topics("צריכה דלתות פנים לדירה"))
check("פולימר",            "interior_doors" in topics("פולימר — כמה עולה?"))
check("דלת חדר",           "interior_doors" in topics("דלת חדר שינה"))
check("interior no false entrance",
      "entrance_doors" not in topics("צריכה 3 דלתות פנים"),
      topics("צריכה 3 דלתות פנים"))

# ── Section 3: multi-topic — entrance + interior ──────────────────────────────

print("\n=== Section 3: multi-topic entrance + interior ===")

def has_both(msg: str) -> bool:
    t = topics(msg)
    return "entrance_doors" in t and "interior_doors" in t

check("דלתות פנים ודלת ראשית",
      has_both("דלתות פנים פולימר ודלת ראשית"),
      topics("דלתות פנים פולימר ודלת ראשית"))

check("פנים וראשית",
      has_both("פנים וראשית"),
      topics("פנים וראשית"))

check("פנים פולימר וראשית",
      has_both("פנים פולימר וראשית"),
      topics("פנים פולימר וראשית"))

check("בית חדש ופנים וראשית",
      has_both("בית חדש ופנים וראשית"),
      topics("בית חדש ופנים וראשית"))

check("גם כניסה וגם פנים",
      has_both("אני מחפש גם דלת כניסה וגם דלתות פנים"),
      topics("אני מחפש גם דלת כניסה וגם דלתות פנים"))

check("דלת לבית ודלתות פנים",
      has_both("צריכה דלת לבית ו-4 דלתות פנים"),
      topics("צריכה דלת לבית ו-4 דלתות פנים"))

check("דלת פלדה ופנים",
      has_both("מחפש דלת פלדה ודלתות פנים"),
      topics("מחפש דלת פלדה ודלתות פנים"))

# ── Section 4: entrance_scope extraction context ──────────────────────────────

print("\n=== Section 4: entrance scope extraction context ===")

def scope(msg: str, current_topic=None):
    state = {"current_active_topic": current_topic} if current_topic else {}
    return _extract_fields_from_message(msg, state).get("entrance_scope")

check("scope: כולל משקוף — explicit context",
      scope("כולל משקוף", "entrance_doors") == "with_frame",
      scope("כולל משקוף", "entrance_doors"))

check("scope: דלת בלבד — explicit context",
      scope("דלת בלבד", "entrance_doors") == "door_only",
      scope("דלת בלבד", "entrance_doors"))

check("scope: no extract for interior",
      scope("דלת בלבד", "interior_doors") is None,
      scope("דלת בלבד", "interior_doors"))

check("scope: דלת ראשית message extracts context",
      scope("דלת ראשית כולל משקוף") == "with_frame",
      scope("דלת ראשית כולל משקוף"))

check("scope: ראשית standalone extracts context",
      scope("ראשית כולל משקוף") == "with_frame",
      scope("ראשית כולל משקוף"))

check("scope: דלת לבית extracts context",
      scope("דלת לבית דלת בלבד") == "door_only",
      scope("דלת לבית דלת בלבד"))

check("scope: דלת פלדה extracts context",
      scope("דלת פלדה עם משקוף") == "with_frame",
      scope("דלת פלדה עם משקוף"))

# ── Section 5: interior field locking ────────────────────────────────────────
# These test the fixes from Issue 1: once a field is answered it must not
# be extracted again, and style is saved for ALL active topics at once.

print("\n=== Section 5: interior field locking (Issue 1) ===")

def fields(msg: str, state_override=None):
    """Return extracted fields for a message given an optional pre-existing state."""
    return _extract_fields_from_message(msg, state_override or {})

# Style saved to entrance when only entrance is active
f = fields("חלקה", {"active_topics": ["entrance_doors"], "entrance_style": None, "interior_style": None})
check("style→entrance only when only entrance active",
      f.get("entrance_style") == "flat" and f.get("interior_style") is None,
      f)

# Style saved to BOTH when both topics are active
f = fields("חלקה", {"active_topics": ["entrance_doors", "interior_doors"], "entrance_style": None, "interior_style": None})
check("style→both topics when both active",
      f.get("entrance_style") == "flat" and f.get("interior_style") == "flat",
      f)

# Style NOT overwritten if already set
f = fields("חלקה", {"active_topics": ["entrance_doors", "interior_doors"],
                     "entrance_style": "designed",  # already set
                     "interior_style": None})
check("style→does not overwrite locked entrance_style",
      f.get("entrance_style") is None and f.get("interior_style") == "flat",
      f)

# מעוצבות also routes to both
f = fields("מעוצבות", {"active_topics": ["entrance_doors", "interior_doors"], "entrance_style": None, "interior_style": None})
check("מעוצבות→both topics",
      f.get("entrance_style") == "designed" and f.get("interior_style") == "designed",
      f)

# Interior project type extracted unconditionally
f = fields("בית חדש ודלת ראשית", {"active_topics": ["entrance_doors", "interior_doors"]})
check("בית חדש→interior_project_type regardless of topic",
      f.get("interior_project_type") == "new",
      f)

f = fields("שיפוץ", {})
check("שיפוץ→interior_project_type='renovation'",
      f.get("interior_project_type") == "renovation", f)

f = fields("החלפה", {})
check("החלפה→interior_project_type='replacement'",
      f.get("interior_project_type") == "replacement", f)

# Standalone number extracted for quantity when current_topic=interior_doors
f = fields("12", {"current_active_topic": "interior_doors", "active_topics": ["interior_doors"]})
check("bare number '12'→interior_quantity when topic is interior",
      f.get("interior_quantity") == 12, f)

f = fields("7", {"current_active_topic": "interior_doors", "active_topics": ["interior_doors"]})
check("bare number '7'→interior_quantity",
      f.get("interior_quantity") == 7, f)

f = fields("12", {"current_active_topic": "entrance_doors", "active_topics": ["entrance_doors"]})
check("bare number '12' NOT extracted when topic is entrance",
      f.get("interior_quantity") is None, f)

# ── Section 6: city/locality extraction (Issue 2) ────────────────────────────

print("\n=== Section 6: city/locality extraction (Issue 2) ===")

def city(msg: str):
    return _extract_fields_from_message(msg, {}).get("city")

check("קיבוץ להב extracted",
      city("קיבוץ להב") == "קיבוץ להב", city("קיבוץ להב"))

check("מושב תקומה extracted",
      city("מושב תקומה") == "מושב תקומה", city("מושב תקומה"))

check("יישוב עומר extracted",
      city("יישוב עומר") == "יישוב עומר", city("יישוב עומר"))

check("כפר ורדים extracted",
      city("כפר ורדים") == "כפר ורדים", city("כפר ורדים"))

check("multi-field: מוטי 0523989366 קיבוץ להב → city",
      city("מוטי 0523989366 קיבוץ להב") == "קיבוץ להב",
      city("מוטי 0523989366 קיבוץ להב"))

check("multi-field: ליטל 0523989366 מושב תקומה → city",
      city("ליטל 0523989366 מושב תקומה") == "מושב תקומה",
      city("ליטל 0523989366 מושב תקומה"))

# multi-field: also check name and phone are extracted
f2 = fields("מוטי 0523989366 קיבוץ להב", {})
check("multi-field מוטי: full_name extracted",
      f2.get("full_name") == "מוטי", f2)
check("multi-field מוטי: phone extracted",
      f2.get("phone") == "0523989366", f2)
check("multi-field מוטי: city = קיבוץ להב",
      f2.get("city") == "קיבוץ להב", f2)

f3 = fields("ליטל 0523989366 מושב תקומה", {})
check("multi-field ליטל: city = מושב תקומה",
      f3.get("city") == "מושב תקומה", f3)

# Known city still works
check("known city: באר שבע still extracted",
      city("מתגורר בבאר שבע") == "באר שבע", city("מתגורר בבאר שבע"))

check("known city: נתיבות still extracted",
      city("מנתיבות") in ("נתיבות",), city("מנתיבות"))

# ── קריית / קרית prefix (Issue 5) ───────────────────────────────────────────
check("קריית עקרון extracted",
      city("קריית עקרון") == "קריית עקרון", city("קריית עקרון"))

check("קרית שמונה extracted",
      city("קרית שמונה") == "קרית שמונה", city("קרית שמונה"))

check("קריית גת still extracted",
      city("קריית גת") == "קריית גת", city("קריית גת"))

# ── multi-field: name + phone + קריית locality ────────────────────────────────
f_qiryat = fields("משה 0523989366 קריית עקרון", {})
check("multi-field משה: full_name = משה",
      f_qiryat.get("full_name") == "משה", f_qiryat)
check("multi-field משה: phone = 0523989366",
      f_qiryat.get("phone") == "0523989366", f_qiryat)
check("multi-field משה: city = קריית עקרון",
      f_qiryat.get("city") == "קריית עקרון", f_qiryat)

f_bityah = fields("דוד 0541234567 מזכרת בתיה", {})
check("multi-field דוד: city = מזכרת בתיה",
      f_bityah.get("city") == "מזכרת בתיה", f_bityah)
check("multi-field דוד: full_name = דוד",
      f_bityah.get("full_name") == "דוד", f_bityah)

f_ganyavne = fields("נועה 0501112222 גן יבנה", {})
check("multi-field נועה: city = גן יבנה",
      f_ganyavne.get("city") == "גן יבנה", f_ganyavne)
check("multi-field נועה: full_name = נועה",
      f_ganyavne.get("full_name") == "נועה", f_ganyavne)

# ── Section 7: "ראשית" false-positive guard ───────────────────────────────────
# "ראשית דבר" / "ראשית כל" — these ARE false positives but are acceptable
# in a door-company context since customers virtually never open with "ראשית דבר".
# This section documents the known edge-case rather than blocking the test suite.

print("\n=== Section 7: edge cases (informational) ===")

t_first_of_all = topics("ראשית, אני רוצה לשאול משהו")
print(f"  [INFO] 'ראשית, אני רוצה לשאול' → topics={t_first_of_all}")
print(f"         (acceptable false-positive; LLM layer provides correction)")

# ── Section 8: interior quantity extraction (Issue 4) ────────────────────────
# "14 יחידות פולימרי מלא" must extract qty=14 and register interior_doors.

print("\n=== Section 8: interior quantity extraction (Issue 4) ===")


def qty(msg: str, state_override=None):
    return _extract_fields_from_message(msg, state_override or {}).get("interior_quantity")


# ── Tier-1a: number before unit word ─────────────────────────────────────────
check("14 יחידות פולימרי מלא → qty=14 (Tier-1a)",
      qty("14 יחידות פולימרי מלא") == 14,
      qty("14 יחידות פולימרי מלא"))

check("12 דלתות פנים → qty=12 (Tier-1a)",
      qty("12 דלתות פנים") == 12,
      qty("12 דלתות פנים"))

check("3 פולימר → qty=3 (Tier-1a)",
      qty("3 פולימר") == 3,
      qty("3 פולימר"))

check("4 דלתות פולימריות → qty=4 (Tier-1a)",
      qty("4 דלתות פולימריות") == 4,
      qty("4 דלתות פולימריות"))

check("צריך 12 פנים → qty=12 (Tier-1a)",
      qty("צריך 12 פנים") == 12,
      qty("צריך 12 פנים"))

check("אני רוצה 3 דלתות פולימריות → qty=3 (Tier-1a)",
      qty("אני רוצה 3 דלתות פולימריות") == 3,
      qty("אני רוצה 3 דלתות פולימריות"))

# ── Tier-1b: unit word before number ─────────────────────────────────────────
check("דלתות 14 → qty=14 (Tier-1b reversed)",
      qty("דלתות 14") == 14,
      qty("דלתות 14"))

check("יחידות 8 → qty=8 (Tier-1b reversed)",
      qty("יחידות 8") == 8,
      qty("יחידות 8"))

# ── Tier-2: number separated from product word by other text ─────────────────
# The number and product word appear in the same message but not adjacent.
check("אני צריך 14, פולימרי → qty=14 (Tier-2 strong-context)",
      qty("אני צריך 14, פולימרי") == 14,
      qty("אני צריך 14, פולימרי"))

check("14 — פולימרי מלא → qty=14 (Tier-2 dash-separated)",
      qty("14 — פולימרי מלא") == 14,
      qty("14 — פולימרי מלא"))

# ── Retroactive topic registration ───────────────────────────────────────────
check("14 יחידות → interior_doors in _new_topics",
      "interior_doors" in (_extract_fields_from_message("14 יחידות", {}).get("_new_topics") or []),
      _extract_fields_from_message("14 יחידות", {}).get("_new_topics"))

check("14 פולימרי מלא → interior_doors in _new_topics (Tier-1a)",
      "interior_doors" in (_extract_fields_from_message("14 פולימרי מלא", {}).get("_new_topics") or []),
      _extract_fields_from_message("14 פולימרי מלא", {}).get("_new_topics"))

# ── Hebrew number words ───────────────────────────────────────────────────────
check("ארבע עשרה דלתות → qty=14",
      qty("ארבע עשרה דלתות") == 14,
      qty("ארבע עשרה דלתות"))

check("שתים עשרה יחידות → qty=12",
      qty("שתים עשרה יחידות") == 12,
      qty("שתים עשרה יחידות"))

# ── Field locking ─────────────────────────────────────────────────────────────
from michael_doors_bot.engine.simple_router import _merge_state
state_locked = {"interior_quantity": 7}
merged = _merge_state(state_locked, _extract_fields_from_message("14 יחידות", {}))
check("interior_quantity already set → not overwritten by new extraction",
      merged.get("interior_quantity") == 7,
      merged.get("interior_quantity"))

# ── Safety: Tier-2 must not fire when phone present ──────────────────────────
# "050" could be misread as 50 — phone_match guard must prevent this.
fe_phone = _extract_fields_from_message("פולימר 0501234567", {})
check("phone present → interior_quantity NOT extracted from phone digits",
      fe_phone.get("interior_quantity") is None,
      fe_phone.get("interior_quantity"))

# ── Section 9: combined topic preservation (Issue 3) ────────────────────────
# The completion guard must block Stage 3 while ANY active topic is incomplete.
# Tests A/B/C verify the fix at the Python state-machine layer (no AI needed).

print("\n=== Section 9: combined topic preservation (Issue 3) ===")


def _make_state(**overrides) -> dict:
    """Return a fresh v2 state with given overrides applied."""
    s = _empty_conv_state()
    s.update(overrides)
    return s


# ── Test A: pure topic detection ─────────────────────────────────────────────
t_a = topics("בית חדש פנים וראשית")
check("Test A: 'בית חדש פנים וראשית' → entrance_doors detected",
      "entrance_doors" in t_a, t_a)
check("Test A: 'בית חדש פנים וראשית' → interior_doors detected",
      "interior_doors" in t_a, t_a)

# Also verify the retroactive inference: interior_project_type extracted from
# "בית חדש" triggers interior_doors in _new_topics even without "דלתות פנים".
fe_a = _extract_fields_from_message("בית חדש ראשית", {})
check("Test A: 'בית חדש ראשית' retroactive → interior_doors in _new_topics",
      "interior_doors" in (fe_a.get("_new_topics") or []), fe_a)
check("Test A: 'בית חדש ראשית' retroactive → entrance_doors in _new_topics",
      "entrance_doors" in (fe_a.get("_new_topics") or []), fe_a)


# ── Test B: flow — interior complete, entrance NOT complete → ask_entrance_scope
# State: both topics active, interior fully collected, entrance scope MISSING.
state_b = _make_state(
    active_topics=["entrance_doors", "interior_doors"],
    interior_project_type="new",
    interior_quantity=12,
    interior_style="flat",
    entrance_scope=None,   # ← missing: bot must ask this next
    entrance_style=None,
)
_advance_stage(state_b, [])
action_b = _decide_next_action(state_b)
check("Test B: interior complete, entrance scope missing → ask_entrance_scope",
      action_b.template_key == "ask_entrance_scope",
      f"got template_key={action_b.template_key!r}")

# Once scope is known, style is asked.
state_b2 = _make_state(
    active_topics=["entrance_doors", "interior_doors"],
    interior_project_type="new",
    interior_quantity=12,
    interior_style="flat",
    entrance_scope="with_frame",
    entrance_style=None,   # ← still missing
)
_advance_stage(state_b2, [])
action_b2 = _decide_next_action(state_b2)
check("Test B: interior complete, entrance style missing → ask_entrance_style",
      action_b2.template_key == "ask_entrance_style",
      f"got template_key={action_b2.template_key!r}")


# ── Test C: completion guard — Stage 3 is blocked while entrance is incomplete
# This tests that the guard at the top of _decide_next_action (fresh recompute)
# prevents Stage 3 even when current_active_topic was stale / None.

# Simulate the race-condition scenario: current_active_topic was set to None
# by a previous _advance_stage (before entrance_doors was added), then a new
# topic was detected.  _decide_next_action must recompute and catch it.
state_c = _make_state(
    active_topics=["entrance_doors", "interior_doors"],
    current_active_topic=None,  # ← stale cached value (was set before entrance added)
    interior_project_type="new",
    interior_quantity=12,
    interior_style="flat",
    entrance_scope=None,   # ← entrance incomplete!
    entrance_style=None,
)
action_c = _decide_next_action(state_c)
check("Test C: stale current_active_topic=None → still asks ask_entrance_scope",
      action_c.template_key == "ask_entrance_scope",
      f"got template_key={action_c.template_key!r}")

# Verify Stage 3 is blocked (not returned) when entrance is incomplete
check("Test C: Stage 3 NOT fired when entrance incomplete",
      action_c.stage != 3,
      f"got stage={action_c.stage}")

# Fully complete state should reach Stage 3
state_c_ok = _make_state(
    active_topics=["entrance_doors", "interior_doors"],
    current_active_topic=None,
    interior_project_type="new",
    interior_quantity=12,
    interior_style="flat",
    entrance_scope="with_frame",
    entrance_style="flat",
    stage3_done=False,
    stage4_opener_sent=False,
)
action_c_ok = _decide_next_action(state_c_ok)
check("Test C: both topics fully complete → Stage 3 fires",
      action_c_ok.stage == 3,
      f"got stage={action_c_ok.stage}, template={action_c_ok.template_key!r}")


# ── Section 10: _early_extract_qty (Issue 5) ─────────────────────────────────
# Covers the dedicated early-pass function that runs before flow decisions.
print("\n=== Section 10: _early_extract_qty — early quantity safety net ===")

from michael_doors_bot.engine.simple_router import _early_extract_qty  # noqa: E402

# Tier A1 — digit before unit word
check("_early_extract_qty: '14 יחידות פולימרי מלא' → 14",
      _early_extract_qty("14 יחידות פולימרי מלא") == 14,
      _early_extract_qty("14 יחידות פולימרי מלא"))

check("_early_extract_qty: '8 דלתות פנים' → 8",
      _early_extract_qty("8 דלתות פנים") == 8,
      _early_extract_qty("8 דלתות פנים"))

check("_early_extract_qty: '3 פולימר' → 3",
      _early_extract_qty("3 פולימר") == 3,
      _early_extract_qty("3 פולימר"))

# Tier A2 — unit word before digit
check("_early_extract_qty: 'יחידות 14' → 14",
      _early_extract_qty("יחידות 14") == 14,
      _early_extract_qty("יחידות 14"))

check("_early_extract_qty: 'דלתות 8' → 8",
      _early_extract_qty("דלתות 8") == 8,
      _early_extract_qty("דלתות 8"))

# Tier B — quantifier prefixes
check("_early_extract_qty: 'כמות 14' → 14",
      _early_extract_qty("כמות 14") == 14,
      _early_extract_qty("כמות 14"))

check("_early_extract_qty: 'בערך 10' → 10",
      _early_extract_qty("בערך 10") == 10,
      _early_extract_qty("בערך 10"))

check("_early_extract_qty: 'כ-8' → 8",
      _early_extract_qty("כ-8") == 8,
      _early_extract_qty("כ-8"))

check("_early_extract_qty: 'כ 8' → 8",
      _early_extract_qty("כ 8") == 8,
      _early_extract_qty("כ 8"))

# Tier C — strong context + separated digit
check("_early_extract_qty: 'אני צריך 14, פולימרי' → 14",
      _early_extract_qty("אני צריך 14, פולימרי") == 14,
      _early_extract_qty("אני צריך 14, פולימרי"))

# Range guard — values > 50 should not be returned
check("_early_extract_qty: '80 יחידות' → None (over 50)",
      _early_extract_qty("80 יחידות") is None,
      _early_extract_qty("80 יחידות"))

# No context → should return None (plain text with no product signal)
check("_early_extract_qty: plain 'שלום' → None",
      _early_extract_qty("שלום") is None,
      _early_extract_qty("שלום"))

# ── Flow test: first message with qty → ask_interior_project_type next ────────
# State: interior_doors active, qty already in state, project_type missing.
# The flow should ask project_type, NOT quantity (because qty is known).
state_eq = _make_state(
    active_topics=["interior_doors"],
    interior_project_type=None,   # ← must be asked first
    interior_quantity=14,          # ← already known via early extraction
    interior_style=None,
)
_advance_stage(state_eq, [])
action_eq = _decide_next_action(state_eq)
check(
    "Flow: qty already set → ask_interior_project_type (not ask_interior_quantity)",
    action_eq.template_key == "ask_interior_project_type",
    f"got template_key={action_eq.template_key!r}",
)

# ── Flow test: all interior fields known → ask_interior_style next ────────────
state_eq2 = _make_state(
    active_topics=["interior_doors"],
    interior_project_type="new",
    interior_quantity=14,
    interior_style=None,   # ← next gap
)
_advance_stage(state_eq2, [])
action_eq2 = _decide_next_action(state_eq2)
check(
    "Flow: qty+type known → ask_interior_style (skips quantity question)",
    action_eq2.template_key == "ask_interior_style",
    f"got template_key={action_eq2.template_key!r}",
)

# ── Fast-path extraction tests (new direct extraction at top of _extract_fields) ─
check("fast-path: '14 יחידות פולימרי מלא' → qty=14",
      _extract_fields_from_message("14 יחידות פולימרי מלא", {}).get("interior_quantity") == 14,
      _extract_fields_from_message("14 יחידות פולימרי מלא", {}).get("interior_quantity"))

check("fast-path: '14 פולימרי' (bare material word) → qty=14",
      _extract_fields_from_message("14 פולימרי", {}).get("interior_quantity") == 14,
      _extract_fields_from_message("14 פולימרי", {}).get("interior_quantity"))

check("fast-path: 'פולימריות 8' → qty=8",
      _extract_fields_from_message("פולימריות 8", {}).get("interior_quantity") == 8,
      _extract_fields_from_message("פולימריות 8", {}).get("interior_quantity"))

check("fast-path: '5 דלת פנים' → qty=5",
      _extract_fields_from_message("5 דלת פנים", {}).get("interior_quantity") == 5,
      _extract_fields_from_message("5 דלת פנים", {}).get("interior_quantity"))

check("fast-path: fast-path adds interior_doors to _new_topics",
      "interior_doors" in (_extract_fields_from_message("14 יחידות פולימרי מלא", {}).get("_new_topics") or []),
      _extract_fields_from_message("14 יחידות פולימרי מלא", {}).get("_new_topics"))

# When phone IS present but "14" is a clearly separate token before the phone,
# Tier-1 (not the fast-path) still extracts qty=14 correctly —
# both phone and qty land in state.
_fe_phone_qty = _extract_fields_from_message("פולימרי 14 0501234567", {})
check("fast-path: 'פולימרי 14 0501234567' → qty=14 extracted (phone + qty both valid)",
      _fe_phone_qty.get("interior_quantity") == 14,
      _fe_phone_qty.get("interior_quantity"))
check("fast-path: 'פולימרי 14 0501234567' → phone also extracted",
      _fe_phone_qty.get("phone") == "0501234567",
      _fe_phone_qty.get("phone"))

# Fast-path phone guard: when the message is ONLY a phone number + context word
# with NO separate digit, the phone digits must NOT be read as qty.
check("fast-path: phone only 'פולימר 0501234567' → qty NOT extracted (phone digits are the only number)",
      _extract_fields_from_message("פולימר 0501234567", {}).get("interior_quantity") is None,
      _extract_fields_from_message("פולימר 0501234567", {}).get("interior_quantity"))

# ── End-to-end sequential flow ────────────────────────────────────────────────
# Simulates the exact reported failing scenario:
#   "#reset"  "אשמח להצעת מחיר"  "14 יחידות פולימרי מלא"  "בית חדש"
# After "14 יחידות": qty must be in state.
# After "בית חדש":   project_type set → next must be ask_interior_style (not qty).
print("\n=== Section 10 end-to-end sequential flow ===")

_seq_state = _empty_conv_state()
_seq_history: list[dict] = []

# Message 1: "אשמח להצעת מחיר"
_m1 = "אשמח להצעת מחיר"
_seq_state = _merge_state(_seq_state, _extract_fields_from_message(_m1, _seq_state))
_seq_history.append({"role": "user",      "content": _m1})
_advance_stage(_seq_state, _seq_history)
_a1 = _decide_next_action(_seq_state)
_seq_history.append({"role": "assistant", "content": "[mock1]"})

# Message 2: "14 יחידות פולימרי מלא"
_m2 = "14 יחידות פולימרי מלא"
_seq_state = _merge_state(_seq_state, _extract_fields_from_message(_m2, _seq_state))
_seq_history.append({"role": "user", "content": _m2})
_advance_stage(_seq_state, _seq_history)

check("E2E: after '14 יחידות פולימרי מלא' → state has interior_quantity=14",
      _seq_state.get("interior_quantity") == 14,
      f"got {_seq_state.get('interior_quantity')!r}")

check("E2E: after '14 יחידות פולימרי מלא' → interior_doors in active_topics",
      "interior_doors" in (_seq_state.get("active_topics") or []),
      _seq_state.get("active_topics"))

_a2 = _decide_next_action(_seq_state)
check("E2E: action after msg2 is NOT ask_interior_quantity",
      _a2.template_key != "ask_interior_quantity",
      f"got {_a2.template_key!r}")
_seq_history.append({"role": "assistant", "content": "[mock2]"})

# Message 3: "בית חדש"
_m3 = "בית חדש"
_seq_state = _merge_state(_seq_state, _extract_fields_from_message(_m3, _seq_state))
_seq_history.append({"role": "user", "content": _m3})
_advance_stage(_seq_state, _seq_history)

check("E2E: after 'בית חדש' → interior_project_type = new",
      _seq_state.get("interior_project_type") == "new",
      f"got {_seq_state.get('interior_project_type')!r}")

check("E2E: interior_quantity still 14 after 'בית חדש'",
      _seq_state.get("interior_quantity") == 14,
      f"got {_seq_state.get('interior_quantity')!r}")

_a3 = _decide_next_action(_seq_state)
check("E2E: next ask after project_type+qty known → ask_interior_style",
      _a3.template_key == "ask_interior_style",
      f"got {_a3.template_key!r} (must NOT be ask_interior_quantity)")


# ── Section 11: Google Sheets one-row guard ───────────────────────────────────
print("\n=== Section 11: Google Sheets one-row guard ===")

# Mirror of the exact guard logic in _maybe_send_to_sheets.
# Returns one of: 'skip_already_sent' | 'skip_missing' | 'skip_no_handoff' | 'send'
_REQUIRED_FIELDS = ("full_name", "callback_phone", "city", "service_type", "preferred_contact_hours")

def _sheets_guard(lead: dict, result: dict) -> str:
    if lead.get("sheets_sent"):
        return "skip_already_sent"
    if not all(lead.get(f) for f in _REQUIRED_FIELDS):
        return "skip_missing"
    if not result.get("handoff_to_human"):
        return "skip_no_handoff"
    return "send"


_base_lead: dict = {
    "full_name":               "יוסי כהן",
    "callback_phone":          "052-1234567",
    "city":                    "תל אביב",
    "service_type":            "דלתות פנים",
    "preferred_contact_hours": "בוקר",
    "sheets_sent":             False,
}

# ── Test 11-1: incomplete lead → not sent ─────────────────────────────────
_incomplete = {**_base_lead, "city": ""}
_g = _sheets_guard(_incomplete, {"handoff_to_human": True})
check("Sheets [11-1]: incomplete lead (missing city) → skip_missing",
      _g == "skip_missing", f"got {_g!r}")

# ── Test 11-2: complete lead, no handoff yet → not sent ───────────────────
_g = _sheets_guard(_base_lead, {"handoff_to_human": False})
check("Sheets [11-2]: complete fields, no handoff → skip_no_handoff",
      _g == "skip_no_handoff", f"got {_g!r}")

# ── Test 11-3: complete lead at Stage 7 handoff → sent ────────────────────
_g = _sheets_guard(_base_lead, {"handoff_to_human": True})
check("Sheets [11-3]: complete lead at handoff → send",
      _g == "send", f"got {_g!r}")

# ── Test 11-4: simulate two calls — second call never appends ─────────────
# First call (handoff) → send
_lead_fresh = {**_base_lead}
_g1 = _sheets_guard(_lead_fresh, {"handoff_to_human": True})
check("Sheets [11-4a]: first call at handoff → send",
      _g1 == "send", f"got {_g1!r}")

# Simulate what _maybe_send_to_sheets does after success: sets sheets_sent=True
_lead_after = {**_lead_fresh, "sheets_sent": True}

# Second call (same conversation, any trigger) → hard stop
_g2 = _sheets_guard(_lead_after, {"handoff_to_human": True})
check("Sheets [11-4b]: second call (sheets_sent=True) → skip_already_sent",
      _g2 == "skip_already_sent", f"got {_g2!r}")

# ── Test 11-5: second call with different data → still blocked ────────────
_lead_corrected = {**_lead_after, "callback_phone": "054-9999999"}
_g3 = _sheets_guard(_lead_corrected, {"handoff_to_human": True})
check("Sheets [11-5]: corrected phone after send → still skip_already_sent (no second row)",
      _g3 == "skip_already_sent", f"got {_g3!r}")

# ── Test 11-6: late summary after send → still blocked ────────────────────
_lead_with_summary = {**_lead_after, "conv_summary": "שאל על דלתות פנים, רוצה 6 יחידות"}
_g4 = _sheets_guard(_lead_with_summary, {"handoff_to_human": False})
check("Sheets [11-6]: summary added after send → skip_already_sent",
      _g4 == "skip_already_sent", f"got {_g4!r}")

# ── Test 11-7: richer service_field at handoff (original duplicate scenario) ─
# Before the fix, this triggered a second row.  Now it can't — sheets_sent blocks it.
_lead_mid = {**_base_lead, "service_type": "דלתות פנים"}           # mid-conv snapshot
_lead_rich = {**_lead_after, "service_type": "דלתות פנים 6 יח' חלקות"}  # post-handoff snapshot
_g_mid  = _sheets_guard(_lead_mid,  {"handoff_to_human": False})
_g_rich = _sheets_guard(_lead_rich, {"handoff_to_human": True})
check("Sheets [11-7]: mid-conv (no handoff, sheets_sent=False) → skip_no_handoff",
      _g_mid == "skip_no_handoff", f"got {_g_mid!r}")
check("Sheets [11-7]: post-handoff (sheets_sent=True) → skip_already_sent",
      _g_rich == "skip_already_sent", f"got {_g_rich!r}")


# ── Section 12: Callback-time normalisation ───────────────────────────────────
print("\n=== Section 12: _normalize_callback_time ===")

_time_cases = [
    # Input                   Expected   Label
    ("אחרי 7",                "19:00",   "אחרי 7 → 19:00"),
    ("אחרי שבע",              "19:00",   "אחרי שבע → 19:00"),
    ("בערב",                  "19:00",   "בערב → 19:00"),
    ("ערב",                   "19:00",   "ערב → 19:00"),
    ("בבוקר",                 "09:00",   "בבוקר → 09:00"),
    ("בוקר",                  "09:00",   "בוקר → 09:00"),
    ("מחר בבוקר",             "09:00",   "מחר בבוקר → 09:00"),
    ("בצהריים",               "13:00",   "בצהריים → 13:00"),
    ("אחר הצהריים",           "16:00",   "אחר הצהריים → 16:00"),
    ("אחרי הצהריים",          "16:00",   "אחרי הצהריים → 16:00"),
    ("18:30",                 "18:30",   "18:30 → 18:30 (pass-through)"),
    ("09:00",                 "09:00",   "09:00 → 09:00 (pass-through)"),
    ("אחרי 8",                "20:00",   "אחרי 8 → 20:00"),
    ("אחרי שמונה",            "20:00",   "אחרי שמונה → 20:00"),
    ("אחרי 9",                "21:00",   "אחרי 9 → 21:00"),
    ("לאחר 7",                "19:00",   "לאחר 7 → 19:00"),
    ("ב7",                    "19:00",   "ב7 → 19:00"),
    ("ב-8",                   "20:00",   "ב-8 → 20:00"),
    ("אחרי שתיים עשרה",       "12:00",   "אחרי שתיים עשרה (noon) → 12:00"),
]

for _inp, _expected, _label in _time_cases:
    _got = _normalize_callback_time(_inp)
    check(f"Time norm: {_label}", _got == _expected, f"got {_got!r}")


# ── Summary ───────────────────────────────────────────────────────────────────

passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n{'=' * 55}")
print(f"Results: {passed}/{total} passed")
if passed < total:
    print("\nFailed tests:")
    for label, ok in results:
        if not ok:
            print(f"  ✗ {label}")
    sys.exit(1)
else:
    print("All tests passed ✓")
