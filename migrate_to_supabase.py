#!/usr/bin/env python3
"""
One-time migration: uploads system prompt + FAQ to Supabase.
Run: .venv/bin/python3 migrate_to_supabase.py
"""
import os, sys, json, asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import httpx

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_KEY", "")

if not URL or not KEY:
    print("[FATAL] SUPABASE_URL or SUPABASE_KEY not set"); sys.exit(1)

HEADERS = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}

BASE = Path(__file__).parent / "src"


async def upsert(table: str, rows: list) -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{URL}/rest/v1/{table}", headers=HEADERS, json=rows)
        r.raise_for_status()
        print(f"  ✅ {table}: {len(rows)} rows uploaded")


async def main():
    print("=== Migrating to Supabase ===\n")

    # 1. System prompt
    print("1. Uploading system prompt...")
    prompt_text = (BASE / "prompts/systemPrompt.txt").read_text(encoding="utf-8")
    await upsert("bot_config", [{"key": "system_prompt", "value": prompt_text}])

    # 2. FAQ bank
    print("2. Uploading FAQ bank...")
    faq = json.loads((BASE / "data/faqBank.json").read_text(encoding="utf-8"))
    rows = [
        {
            "id":       entry["id"],
            "category": entry.get("category", ""),
            "keywords": entry.get("keywords", []),
            "answer":   entry.get("answer", ""),
        }
        for entry in faq
    ]
    await upsert("faq_entries", rows)

    print("\n=== Done! ===")
    print("Check Supabase Table Editor to verify the data.")


if __name__ == "__main__":
    asyncio.run(main())
