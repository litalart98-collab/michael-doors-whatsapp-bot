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
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Credentials (env only) ────────────────────────────────────────────────────
_TOKEN    = os.getenv("AIRTABLE_TOKEN", "").strip()
_BASE_ID  = os.getenv("AIRTABLE_BASE_ID", "").strip()
_TABLE_ID = os.getenv("AIRTABLE_TABLE_ID", "").strip()

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
FIELD_MAP: dict[str, str] = {
    "whatsapp_id":  "WhatsApp ID",
    "phone":        "טלפון",
    "full_name":    "שם מלא",
    "city":         "עיר",
    "topic":        "נושא הפנייה",
    "frame":        "משקוף",
    "doors_count":  "כמות דלתות",
    "project_type": "סוג פרויקט",
    "notes":        "הערות",
    "stage":        "שלב נוכחי",
    "status":       "סטטוס",
    "created_at":   "תאריך פנייה",
    "updated_at":   "תאריך עדכון אחרון",
    "completed_at": "תאריך סיום",
    "summary":      "סיכום",
}


def enabled() -> bool:
    """True only when all three credentials are present."""
    return bool(_TOKEN and _BASE_ID and _TABLE_ID)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_TOKEN}",
        "Content-Type":  "application/json",
    }


def _table_url() -> str:
    # httpx URL-encodes the table segment for us when it contains spaces/Hebrew.
    return f"{_API_ROOT}/{_BASE_ID}/{_TABLE_ID}"


def _clean_fields(values: dict) -> dict:
    """Map internal keys → Airtable columns and drop empty / None values.

    Rules:
      • Unknown internal keys are ignored (defensive).
      • None is dropped.
      • Empty / whitespace-only strings are dropped (never blank out a value).
      • Everything else (numbers, non-empty strings, bools) passes through.
    """
    fields: dict = {}
    for key, value in values.items():
        column = FIELD_MAP.get(key)
        if not column:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        fields[column] = value
    return fields


def _escape_formula_value(raw: str) -> str:
    """Escape a value for safe use inside an Airtable filterByFormula string.

    Values are wrapped in single quotes in the formula, so any single quote in
    the value must be escaped. WhatsApp ids are digits + '@c.us' so this is
    mostly defensive.
    """
    return str(raw).replace("\\", "\\\\").replace("'", "\\'")


# ── Public API ────────────────────────────────────────────────────────────────
async def find_active_lead_by_whatsapp_id(whatsapp_id: str) -> Optional[str]:
    """Return the record id of the ACTIVE inquiry for this WhatsApp id, or None.

    "Active" = status is STATUS_COLLECTING. Used to recover the record id after a
    server restart wiped the in-memory / on-disk mapping, so we UPDATE the open
    record instead of creating a duplicate. Completed inquiries (STATUS_WAITING)
    are intentionally NOT matched, so a returning customer opens a fresh record.
    """
    if not enabled():
        return None
    formula = (
        f"AND("
        f"{{{FIELD_MAP['whatsapp_id']}}}='{_escape_formula_value(whatsapp_id)}',"
        f"{{{FIELD_MAP['status']}}}='{STATUS_COLLECTING}'"
        f")"
    )
    params = {"filterByFormula": formula, "maxRecords": "1"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(_table_url(), headers=_headers(), params=params)
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


async def create_lead(values: dict) -> Optional[str]:
    """Create a new inquiry record. Returns the new record id, or None on failure.

    `values` uses internal keys (see FIELD_MAP). The caller is expected to set at
    least whatsapp_id, phone, created_at and status.
    """
    if not enabled():
        return None
    fields = _clean_fields(values)
    if not fields:
        return None
    payload = {"fields": fields, "typecast": True}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(_table_url(), headers=_headers(), json=payload)
        if r.status_code // 100 != 2:
            logger.warning(
                "[AIRTABLE] create_lead non-2xx | status=%d | body=%s",
                r.status_code, r.text[:300],
            )
            return None
        return r.json().get("id")
    except Exception as exc:
        logger.warning("[AIRTABLE] create_lead failed: %s", exc)
        return None


async def update_lead(record_id: str, values: dict) -> bool:
    """PATCH an existing record with the (non-empty) values. Returns success."""
    if not enabled() or not record_id:
        return False
    fields = _clean_fields(values)
    if not fields:
        return False
    payload = {"fields": fields, "typecast": True}
    url = f"{_table_url()}/{record_id}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.patch(url, headers=_headers(), json=payload)
        if r.status_code // 100 != 2:
            logger.warning(
                "[AIRTABLE] update_lead non-2xx | id=%s | status=%d | body=%s",
                record_id, r.status_code, r.text[:300],
            )
            return False
        return True
    except Exception as exc:
        logger.warning("[AIRTABLE] update_lead failed | id=%s | %s", record_id, exc)
        return False


async def complete_lead(record_id: str, values: dict) -> bool:
    """Mark a record complete: status → STATUS_WAITING plus any final values.

    The caller passes completed_at / summary inside `values`; this helper just
    forces the status so completion always sets it correctly.
    """
    if not enabled() or not record_id:
        return False
    final = dict(values)
    final["status"] = STATUS_WAITING
    return await update_lead(record_id, final)
