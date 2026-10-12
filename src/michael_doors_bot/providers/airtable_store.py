"""
Direct Airtable Web API integration (no Make / n8n).

This module is the SINGLE place that talks HTTP to Airtable and the SINGLE
place that maps the bot's internal field names to the Airtable column names.
Nothing else in the codebase should hard-code an Airtable column name or issue
an Airtable HTTP request.

Credentials come exclusively from environment variables — never hard-coded:

    AIRTABLE_TOKEN      Personal-access token (starts with "pat...")
    AIRTABLE_BASE_ID    Base id (starts with "app...")
    AIRTABLE_TABLE_ID   Table id (starts with "tbl...") or the table name

Design goals (see task spec):
  • Bot keeps working even when Airtable is down — every call is wrapped in
    try/except, has a request timeout, and NEVER raises to the caller.
  • The Airtable token is never written to the logs.
  • Empty / None values are dropped so we never overwrite a real value with a
    blank one, and never send undefined fields.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Credentials ───────────────────────────────────────────────────────────────
# The base/table ids are NOT secret (they appear in the Airtable URL), so they
# default to the customer's known values — the operator only has to set the
# secret AIRTABLE_TOKEN in Render. An env var still overrides the default.
_DEFAULT_BASE_ID  = "appguh7VFfypJmyba"   # "ניהול לידים" base
_DEFAULT_TABLE_ID = "tblfxdYQidDkoG94R"   # Table 1

_TOKEN         = os.getenv("AIRTABLE_TOKEN", "").strip()
_BASE_ID       = os.getenv("AIRTABLE_BASE_ID", "").strip() or _DEFAULT_BASE_ID
_TABLE_ID      = os.getenv("AIRTABLE_TABLE_ID", "").strip() or _DEFAULT_TABLE_ID
# Optional separate table for the browser test simulator (/test-ui). When set,
# test conversations are written here instead of the customer's real table, so
# the production table stays clean. Falls back to empty (test writes disabled).
_TEST_TABLE_ID = os.getenv("AIRTABLE_TEST_TABLE_ID", "").strip()

_API_ROOT = "https://api.airtable.com/v0"
_TIMEOUT  = 8.0  # seconds — keep short so a slow Airtable never stalls WhatsApp

# ── Status values ─────────────────────────────────────────────────────────────
# Kept as constants so main.py sets them symbolically instead of repeating the
# Hebrew strings across the codebase.
STATUS_COLLECTING = "בתהליך איסוף פרטים"   # inquiry open — still collecting details
STATUS_WAITING    = "ממתין לנציג"          # details confirmed — waiting for a human rep

# ── Field mapping: internal bot key  →  Airtable column name ──────────────────
# This is the ONE authoritative mapping. To rename an Airtable column, change it
# here only. Internal keys are stable; Airtable column names are the values.
#
# These MUST match the customer's actual Airtable columns character-for-character
# (including spaces, and the single/double yud of פניה) — Airtable rejects the
# whole request on an unknown column name. Verified against the live table:
#   שם לקוח (text) · עיר (text) · טלפון (phone) · סוג השירות (MULTIPLE select) ·
#   נושא פניה (text) · תאריך פניה (text) · סטטוס (single select)
FIELD_MAP: dict[str, str] = {
    "full_name":  "שם לקוח",
    "city":       "עיר",
    "phone":      "טלפון",
    "service":    "סוג השירות",  # Multiple Select — values are sent as an array
    "topic":      "נושא פנייה",   # best-guess spelling; self-heal drops it if wrong
    "created_at": "תאריך פנייה",
    "status":     "סטטוס",
}

# Internal keys whose Airtable column is a Multiple-Select field: the API needs an
# array of option names, not a plain string. typecast creates missing options.
_MULTI_SELECT_FIELDS = {"service"}

# Accepted column-name VARIANTS per internal key, tried in order until one matches
# the real column (handles single/double yud, trailing/leading space, optional ה).
# The writer auto-adapts to whichever variant the table actually uses, so the code
# never needs the exact spelling. The first entry should match FIELD_MAP.
FIELD_ALIASES: dict[str, list[str]] = {
    "topic": [
        "נושא פנייה", "נושא פניה",
        "נושא פנייה ", "נושא פניה ",
        " נושא פנייה", " נושא פניה",
        "נושא הפנייה", "נושא הפניה",
    ],
    "created_at": [
        "תאריך פנייה", "תאריך פניה",
        "תאריך פנייה ", "תאריך פניה ",
        " תאריך פנייה", " תאריך פניה",
        "תאריך הפנייה", "תאריך הפניה",
    ],
}


def _alias_list(key: str) -> list:
    """All accepted column names for an internal key (its FIELD_MAP value first)."""
    return FIELD_ALIASES.get(key) or [FIELD_MAP[key]]


def enabled() -> bool:
    """True only when all three credentials are present."""
    return bool(_TOKEN and _BASE_ID and _TABLE_ID)


def test_table_id() -> str:
    """The separate test table id, or '' if AIRTABLE_TEST_TABLE_ID is not set."""
    return _TEST_TABLE_ID


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_TOKEN}",
        "Content-Type":  "application/json",
    }


def _table_url(table_id: Optional[str] = None) -> str:
    # httpx URL-encodes the table segment for us when it contains spaces/Hebrew.
    return f"{_API_ROOT}/{_BASE_ID}/{table_id or _TABLE_ID}"


def _clean_pairs(values: dict) -> list:
    """Normalise raw values → a list of (internal_key, value) pairs, dropping
    empty / None values. Column-name mapping is deferred to the writer so it can
    try alias variants per key.

    Rules:
      • Unknown internal keys are ignored (defensive).
      • None is dropped.
      • Empty / whitespace-only strings are dropped (never blank out a value).
      • Multiple-Select values are split on "|" into an array of option names.
      • Everything else (numbers, non-empty strings, bools) passes through.
    """
    pairs: list = []
    for key, value in values.items():
        if key not in FIELD_MAP:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        # Multiple-Select columns need an array of option names. Split a combined
        # "A | B" string into separate options; a single value becomes a 1-element list.
        if key in _MULTI_SELECT_FIELDS and isinstance(value, str):
            opts = [v.strip() for v in value.split("|") if v.strip()]
            if not opts:
                continue
            value = opts
        pairs.append((key, value))
    return pairs


def _clean_fields(values: dict) -> dict:
    """Map internal keys → Airtable columns (first/default alias) and drop empties."""
    return {FIELD_MAP[key]: value for key, value in _clean_pairs(values)}


def preview_fields(values: dict) -> dict:
    """Public helper for dry-run preview: map internal keys → Airtable columns and
    drop empties, WITHOUT sending anything. Returns exactly what would be written."""
    return _clean_fields(values)


def _escape_formula_value(raw: str) -> str:
    """Escape a value for safe use inside an Airtable filterByFormula string.

    Values are wrapped in single quotes in the formula, so any single quote in
    the value must be escaped. WhatsApp ids are digits + '@c.us' so this is
    mostly defensive.
    """
    return str(raw).replace("\\", "\\\\").replace("'", "\\'")


# ── Public API ────────────────────────────────────────────────────────────────
async def find_active_lead_by_phone(phone: str, table_id: Optional[str] = None) -> Optional[str]:
    """Return the record id of the ACTIVE inquiry for this phone number, or None.

    "Active" = status is STATUS_COLLECTING. Used to recover the record id after a
    server restart wiped the in-memory / on-disk mapping, so we UPDATE the open
    record instead of creating a duplicate. Completed inquiries (STATUS_WAITING)
    are intentionally NOT matched, so a returning customer opens a fresh record.

    The customer's table has no dedicated WhatsApp-id column, so we match on the
    phone number (the closest stable per-customer key available).
    """
    if not enabled() or not phone:
        return None
    formula = (
        f"AND("
        f"{{{FIELD_MAP['phone']}}}='{_escape_formula_value(phone)}',"
        f"{{{FIELD_MAP['status']}}}='{STATUS_COLLECTING}'"
        f")"
    )
    params = {"filterByFormula": formula, "maxRecords": "1"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(_table_url(table_id), headers=_headers(), params=params)
        if r.status_code // 100 != 2:
            logger.warning(
                "[AIRTABLE] find_active_lead non-2xx | status=%d | body=%s",
                r.status_code, r.text[:300],
            )
            return None
        records = r.json().get("records", [])
        return records[0]["id"] if records else None
    except Exception as exc:
        logger.warning("[AIRTABLE] find_active_lead_by_whatsapp_id failed: %s", exc)
        return None


async def create_lead(values: dict, table_id: Optional[str] = None) -> Optional[str]:
    """Create a new inquiry record. Returns the new record id, or None on failure.
    Self-heals around unknown column names (see _send_healing)."""
    if not enabled():
        return None
    res = await _send_healing("POST", _table_url(table_id), values)
    if res["dropped"]:
        logger.warning("[AIRTABLE] create skipped unknown columns: %s", res["dropped"])
    if not res["ok"]:
        logger.warning("[AIRTABLE] create_lead failed | status=%s | %s", res["status"], res["error"])
    return res["id"] if res["ok"] else None


async def update_lead(record_id: str, values: dict, table_id: Optional[str] = None) -> bool:
    """PATCH an existing record with the (non-empty) values. Returns success.
    Self-heals around unknown column names."""
    if not enabled() or not record_id:
        return False
    res = await _send_healing("PATCH", f"{_table_url(table_id)}/{record_id}", values)
    if res["dropped"]:
        logger.warning("[AIRTABLE] update skipped unknown columns: %s", res["dropped"])
    return res["ok"]


async def complete_lead(record_id: str, values: dict, table_id: Optional[str] = None) -> bool:
    """Mark a record complete: status → STATUS_WAITING plus any final values."""
    if not enabled() or not record_id:
        return False
    final = dict(values)
    final["status"] = STATUS_WAITING
    return await update_lead(record_id, final, table_id)


# Extracts the offending column from an Airtable 422 "Unknown field name: "X"".
_UNKNOWN_FIELD_RE = re.compile(r'Unknown field name:\s*\\?"([^"\\]+)')


def _build_fields(pairs: list, alias_idx: dict, dropped_keys: set) -> dict:
    """Map (key, value) pairs → Airtable columns using each key's CURRENT alias
    index, skipping keys that have been dropped (all aliases exhausted)."""
    fields: dict = {}
    for key, value in pairs:
        if key in dropped_keys:
            continue
        aliases = _alias_list(key)
        idx = alias_idx.get(key, 0)
        if idx >= len(aliases):
            continue
        fields[aliases[idx]] = value
    return fields


async def _send_healing(method: str, url: str, values: dict) -> dict:
    """Map values→columns and POST/PATCH with typecast. On a 422 UNKNOWN_FIELD_NAME,
    AUTO-ADAPT to the table's real column name: retry the SAME value under the next
    spelling variant for that field (single/double yud, trailing space, optional ה)
    until one matches. Only when every variant for a field is rejected is that one
    field dropped — a column-name mismatch never fails the whole write; the record
    is saved with the columns that DO exist.
    Returns {ok, status, id, error, dropped}."""
    pairs = _clean_pairs(values)
    if not pairs:
        return {"ok": False, "status": None, "id": None, "error": "no fields", "dropped": []}

    alias_idx: dict = {}      # internal key → index into its alias list
    dropped_keys: set = set() # keys whose every alias Airtable rejected
    dropped_cols: list = []   # the actual column names we gave up on (for reporting)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for _ in range(40):
                fields = _build_fields(pairs, alias_idx, dropped_keys)
                if not fields:
                    return {"ok": False, "status": None, "id": None,
                            "error": "no mappable fields", "dropped": dropped_cols}
                r = await client.request(
                    method, url, headers=_headers(),
                    json={"fields": fields, "typecast": True},
                )
                if r.status_code // 100 == 2:
                    try:
                        rid = r.json().get("id")
                    except Exception:
                        rid = None
                    return {"ok": True, "status": r.status_code, "id": rid,
                            "error": None, "dropped": dropped_cols}
                body = r.text
                if r.status_code == 422 and "UNKNOWN_FIELD_NAME" in body:
                    m = _UNKNOWN_FIELD_RE.search(body)
                    # Do NOT strip — a trailing/leading space can be part of the
                    # real column name (that's exactly what we're matching against).
                    bad = m.group(1) if m else None
                    # Reverse-map the rejected column back to its internal key.
                    bad_key = next(
                        (k for k, _v in pairs
                         if k not in dropped_keys
                         and _alias_list(k)[alias_idx.get(k, 0):alias_idx.get(k, 0) + 1] == [bad]),
                        None,
                    )
                    if bad_key is not None:
                        # Try the NEXT spelling variant for this field; if none left, drop it.
                        alias_idx[bad_key] = alias_idx.get(bad_key, 0) + 1
                        if alias_idx[bad_key] >= len(_alias_list(bad_key)):
                            dropped_keys.add(bad_key)
                            dropped_cols.append(bad)
                        continue
                return {"ok": False, "status": r.status_code, "id": None,
                        "error": body[:400], "dropped": dropped_cols}
    except Exception as exc:
        return {"ok": False, "status": None, "id": None, "error": str(exc)[:400], "dropped": dropped_cols}
    return {"ok": False, "status": None, "id": None, "error": "too many unknown fields", "dropped": dropped_cols}


async def list_record_fields(table_id: Optional[str] = None) -> list:
    """Best-effort: return field names seen across up to 50 records (columns that
    are empty in every record won't appear). Diagnostics only."""
    if not enabled():
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(_table_url(table_id), headers=_headers(),
                                  params={"maxRecords": "50"})
        if r.status_code // 100 != 2:
            return []
        names: set = set()
        for rec in r.json().get("records", []):
            names.update((rec.get("fields") or {}).keys())
        return sorted(names)
    except Exception:
        return []


async def selftest_create(values: dict, table_id: Optional[str] = None) -> dict:
    """Diagnostic create used by /test-airtable. Returns detailed status (including
    any self-healed/dropped column names) without exposing the token."""
    missing = [name for name, val in (
        ("AIRTABLE_TOKEN", _TOKEN), ("AIRTABLE_BASE_ID", _BASE_ID),
        ("AIRTABLE_TABLE_ID", _TABLE_ID),
    ) if not val]
    if missing:
        return {"ok": False, "status": None, "id": None,
                "error": "missing env vars", "missing_env": missing}
    res = await _send_healing("POST", _table_url(table_id), values)
    return {
        "ok": res["ok"],
        "status": res["status"],
        "id": res["id"],
        "error": res["error"],
        "missing_env": [],
        "dropped_columns": res["dropped"],
    }

