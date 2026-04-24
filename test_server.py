#!/usr/bin/env python3
"""
Standalone test server — uses the EXACT same get_reply as production.
Run:  python3 test_server.py
Open: http://localhost:3001/test-ui
"""
import os, sys, asyncio, logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

os.environ.setdefault("GREEN_API_INSTANCE_ID", "test")
os.environ.setdefault("GREEN_API_TOKEN", "test")
os.environ.setdefault("TEST_MODE", "true")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json as _json

from michael_doors_bot.engine.simple_router import get_reply, clear_conversation
from michael_doors_bot.providers.google_sheets import append_lead
from test_scenarios import SCENARIOS, run_scenario, ScenarioResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_KEY          = os.environ.get("ANTHROPIC_API_KEY", "")
SHEETS_URL       = os.environ.get("GOOGLE_SHEETS_WEBHOOK_URL", "")
_sheets_sent: set[str] = set()   # senders already sent to Sheets this session

if not API_KEY:
    print("[FATAL] ANTHROPIC_API_KEY not set in .env"); sys.exit(1)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _try_send_sheets(sender: str, result: dict) -> None:
    if not SHEETS_URL:
        logger.warning("[SHEETS] GOOGLE_SHEETS_WEBHOOK_URL not set — skipping")
        return
    if not result.get("handoff_to_human"):
        return
    if sender in _sheets_sent:
        logger.info("[SHEETS] already sent for %s — skipping duplicate", sender)
        return
    # Normalize phone: "972529330102@c.us" → "052-9330102"
    raw = (result.get("phone") or sender).replace("@c.us", "").strip()
    if raw.startswith("972") and len(raw) >= 11:
        raw = "0" + raw[3:]
    digits = raw.replace("-", "").replace(" ", "")
    phone = (digits[:3] + "-" + digits[3:]) if (len(digits) == 10 and digits.isdigit()) else raw
    svc = result.get("service_type") or ""
    cnt = result.get("doors_count")
    service_field = f"{svc} — {cnt} יחידות" if svc and cnt else svc
    row = {
        "full_name":               result.get("full_name") or "",
        "city":                    result.get("city") or "",
        "service_type":            service_field,
        "preferred_contact_hours": result.get("preferred_contact_hours") or "",
        "phone":                   phone,
        "datetime":                __import__("datetime").datetime.utcnow().isoformat(),
    }
    try:
        await append_lead(SHEETS_URL, row)
        _sheets_sent.add(sender)
        logger.info("[SHEETS] Lead sent | phone=%s", phone)
    except Exception as exc:
        logger.error("[SHEETS] Failed to send lead | phone=%s | %s", phone, exc)


@app.post("/test-chat")
async def test_chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    sender  = (data.get("sender") or "test_ui@c.us").strip()
    message = (data.get("message") or "").strip()
    reset   = bool(data.get("reset", False))
    if reset:
        clear_conversation(sender)
        _sheets_sent.discard(sender)
        return JSONResponse({"ok": True, "reset": True})

    if not message:
        return JSONResponse({"ok": False, "error": "empty message"}, status_code=400)

    result = await get_reply(sender, message, API_KEY, mock_claude=False)
    await _try_send_sheets(sender, result)
    return JSONResponse({"ok": True, **result})


@app.get("/test-ui", response_class=HTMLResponse)
async def test_ui():
    return HTMLResponse(_HTML)


@app.get("/run-tests/list")
async def run_tests_list():
    """Return list of all scenario IDs and names."""
    return JSONResponse([{"id": s.id, "name": s.name} for s in SCENARIOS])


@app.post("/run-tests/scenario/{scenario_id}")
async def run_single_test(scenario_id: str):
    """Run one scenario and return the result."""
    scenario = next((s for s in SCENARIOS if s.id == scenario_id), None)
    if not scenario:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        result: ScenarioResult = await asyncio.wait_for(
            run_scenario(scenario, get_reply, API_KEY, clear_conversation),
            timeout=180,
        )
    except asyncio.TimeoutError:
        logger.error("Scenario timeout: %s", scenario_id)
        result = ScenarioResult(scenario=scenario, passed=False, error="timeout after 180s")
    except Exception as e:
        logger.error("run_scenario crash %s: %s", scenario_id, e)
        result = ScenarioResult(scenario=scenario, passed=False, error=str(e))

    return JSONResponse({
        "id": scenario.id,
        "name": scenario.name,
        "passed": result.passed,
        "duration_ms": result.duration_ms,
        "error": result.error,
        "steps": [
            {
                "turn": s.turn,
                "message": s.message,
                "reply": s.reply,
                "passed": s.passed,
                "failures": s.failures,
                "data": {k: v for k, v in s.data.items()
                         if v is not None and v is not False and v != ""},
            }
            for s in result.steps
        ],
    })


@app.get("/test-suite", response_class=HTMLResponse)
async def test_suite():
    return HTMLResponse(_SUITE_HTML)


_HTML = """<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>בוט דלתות מיכאל — בדיקות</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#e5ddd5;height:100vh;display:flex;flex-direction:column;overflow:hidden}
#header{background:#075e54;color:#fff;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0;flex-wrap:wrap}
#header h1{font-size:16px;font-weight:600;flex:1;min-width:160px}
.hctrl{display:flex;align-items:center;gap:6px;font-size:13px}
#sender-input{padding:4px 8px;border-radius:4px;border:none;font-size:13px;width:140px;direction:ltr}
#btn-reset{background:#128c7e;color:#fff;border:none;padding:5px 11px;border-radius:4px;cursor:pointer;font-size:13px}
#btn-reset:hover{background:#0d6f63}
#chat{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:5px}
.msg-row{display:flex;max-width:78%}
.msg-row.user{align-self:flex-start;flex-direction:row-reverse}
.msg-row.bot{align-self:flex-end}
.bubble{padding:8px 12px;border-radius:8px;font-size:14px;line-height:1.55;white-space:pre-wrap;word-break:break-word;box-shadow:0 1px 2px rgba(0,0,0,.15)}
.user .bubble{background:#dcf8c6;border-bottom-left-radius:2px}
.bot  .bubble{background:#fff;border-bottom-right-radius:2px}
.bubble.scripted{background:#f0fff4;border-right:3px solid #4caf50}
.bubble.claude  {background:#fff;border-right:3px solid #2196f3}
.bubble.handoff {background:#e8f4fd;border-right:3px solid #1565c0}
.msg-time{font-size:10px;color:#999;margin-top:2px;text-align:left}
.badge{font-size:10px;padding:1px 6px;border-radius:10px;margin-bottom:3px;display:inline-block}
.b-scripted{background:#c8e6c9;color:#1b5e20}
.b-claude  {background:#e3f2fd;color:#0d47a1}
.b-handoff {background:#bbdefb;color:#0d47a1;font-weight:700}
#meta{background:#f7f7f7;border-top:1px solid #ddd;padding:5px 16px;font-size:11px;color:#555;direction:ltr;min-height:24px;font-family:monospace;flex-shrink:0;overflow-x:auto;white-space:nowrap}
#input-area{background:#f0f0f0;padding:8px 12px;display:flex;gap:8px;align-items:flex-end;flex-shrink:0}
#msg-input{flex:1;padding:10px 14px;border-radius:20px;border:none;font-size:14px;font-family:inherit;resize:none;max-height:120px;outline:none;direction:rtl;line-height:1.4}
#btn-send{background:#075e54;color:#fff;border:none;border-radius:50%;width:44px;height:44px;cursor:pointer;font-size:20px;flex-shrink:0;display:flex;align-items:center;justify-content:center}
#btn-send:hover{background:#128c7e}
#btn-send:disabled{background:#aaa;cursor:not-allowed}
.typing-row{align-self:flex-end}
.typing-bub{background:#fff;border-radius:8px;padding:8px 14px;font-size:13px;color:#888;font-style:italic;box-shadow:0 1px 2px rgba(0,0,0,.1)}
.sys{text-align:center;font-size:11px;color:#888;background:rgba(255,255,255,.6);padding:3px 12px;border-radius:10px;align-self:center;margin:2px 0}
</style>
</head>
<body>
<div id="header">
  <h1>🚪 בוט דלתות מיכאל — ממשק בדיקות</h1>
  <a href="/test-suite" style="color:#d0ede9;font-size:12px;text-decoration:none">🧪 טסטים אוטומטיים</a>
  <div class="hctrl">
    <span>לקוח:</span>
    <input id="sender-input" value="test_1" placeholder="test_1">
  </div>
  <button id="btn-reset">🔄 איפוס שיחה</button>
</div>
<div id="chat"></div>
<div id="meta">מוכן — תגובות זהות לפרודקשן ✓</div>
<div id="input-area">
  <textarea id="msg-input" rows="1" placeholder="כתוב הודעה... (Enter לשליחה)"></textarea>
  <button id="btn-send">➤</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('msg-input'),
      btn=document.getElementById('btn-send'),meta=document.getElementById('meta');

const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const ts=()=>new Date().toLocaleTimeString('he-IL',{hour:'2-digit',minute:'2-digit'});
const sender=()=>(document.getElementById('sender-input').value.trim()||'test_1').replace(/[^\\w]/g,'_')+'@c.us';

function scrollDown(){chat.scrollTop=chat.scrollHeight;}

function sysMsg(t){
  const d=document.createElement('div');d.className='sys';d.textContent=t;
  chat.appendChild(d);scrollDown();
}

function addMsg(text,side,type=''){
  if(!text)return;
  const row=document.createElement('div');row.className='msg-row '+side;
  const badges={scripted:'<span class="badge b-scripted">📋 תסריט</span>',
                 claude:  '<span class="badge b-claude">✨ קלוד</span>',
                 handoff: '<span class="badge b-handoff">✅ הועבר לנציג</span>'};
  row.innerHTML='<div>'+(badges[type]||'')+'<div class="bubble '+type+'">'+esc(text)+'</div><div class="msg-time">'+ts()+'</div></div>';
  chat.appendChild(row);scrollDown();
}

function setMeta(d){
  const p=[];
  if(d.summary)p.push('📝 '+d.summary);
  if(d.service_type)p.push('🚪 '+d.service_type);
  if(d.city)p.push('📍 '+d.city);
  if(d.full_name)p.push('👤 '+d.full_name);
  if(d.phone)p.push('📞 '+d.phone);
  if(d.preferred_contact_hours)p.push('⏰ '+d.preferred_contact_hours);
  if(d.handoff_to_human)p.push('✅ HANDOFF');
  meta.textContent=p.length?p.join('  |  '):'—';
}

async function send(){
  const text=inp.value.trim();if(!text||btn.disabled)return;
  inp.value='';inp.style.height='';btn.disabled=true;
  addMsg(text,'user');
  const ty=document.createElement('div');ty.className='msg-row bot typing-row';
  ty.innerHTML='<div class="typing-bub">נטלי מקלידה...</div>';
  chat.appendChild(ty);scrollDown();
  try{
    const r=await fetch('/test-chat',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sender:sender(),message:text})});
    const d=await r.json();ty.remove();
    if(!d.ok){meta.textContent='שגיאה: '+(d.error||r.status);return;}
    const type=d.handoff_to_human?'handoff':(d.reply_text_2!=null?'scripted':'claude');
    addMsg(d.reply_text,'bot',type);
    if(d.reply_text_2){
      setTimeout(()=>addMsg(d.reply_text_2,'bot','scripted'),450);
    }
    setMeta(d);
    if(d.handoff_to_human)sysMsg('✅ כל הפרטים נאספו — הועבר לנציג');
  }catch(e){ty.remove();meta.textContent='שגיאת רשת: '+e.message;}
  finally{btn.disabled=false;inp.focus();}
}

document.getElementById('btn-reset').onclick=async()=>{
  await fetch('/test-chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sender:sender(),reset:true})});
  chat.innerHTML='';meta.textContent='שיחה אופסה';sysMsg('שיחה חדשה התחילה');
};

btn.onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
inp.addEventListener('input',()=>{inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,120)+'px';});
sysMsg('ממשק בדיקות — תגובות זהות לפרודקשן 🔒');
</script>
</body>
</html>"""


_SUITE_HTML = """<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>בוט מיכאל — חבילת טסטים</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;min-height:100vh;direction:rtl}
#header{background:#075e54;color:#fff;padding:14px 20px;display:flex;align-items:center;gap:12px}
#header h1{font-size:17px;font-weight:600;flex:1}
#btn-run{background:#25d366;color:#fff;border:none;padding:9px 22px;border-radius:6px;cursor:pointer;font-size:14px;font-weight:600}
#btn-run:disabled{background:#888;cursor:not-allowed}
#btn-run:hover:not(:disabled){background:#1ebe5d}
#summary{background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;font-size:13px;color:#444;min-height:38px;display:flex;align-items:center;gap:16px}
.sum-chip{padding:4px 12px;border-radius:12px;font-weight:600;font-size:12px}
.chip-pass{background:#d4edda;color:#155724}
.chip-fail{background:#f8d7da;color:#721c24}
.chip-pend{background:#e2e3e5;color:#383d41}
#scenarios{padding:16px 20px;display:flex;flex-direction:column;gap:10px;max-width:900px;margin:0 auto}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1);overflow:hidden;transition:box-shadow .15s}
.card:hover{box-shadow:0 2px 8px rgba(0,0,0,.15)}
.card-header{padding:12px 16px;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.card-header h2{font-size:14px;font-weight:600;flex:1}
.card-header .desc{font-size:12px;color:#888}
.status-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.dot-pass{background:#28a745}
.dot-fail{background:#dc3545}
.dot-pend{background:#adb5bd}
.dot-running{background:#ffc107;animation:pulse .6s infinite alternate}
@keyframes pulse{from{opacity:.4}to{opacity:1}}
.dur{font-size:11px;color:#aaa;flex-shrink:0}
.card-body{display:none;border-top:1px solid #f0f0f0;padding:12px 16px}
.card-body.open{display:block}
.step{display:flex;gap:10px;margin-bottom:8px;font-size:13px;align-items:flex-start}
.step-num{background:#e9ecef;border-radius:4px;padding:1px 6px;font-size:11px;font-weight:600;color:#555;flex-shrink:0;margin-top:2px}
.step-content{flex:1}
.step-user{color:#075e54;font-weight:500}
.step-bot{color:#333;margin-top:3px;white-space:pre-wrap}
.step-fail{background:#fff3cd;border-right:3px solid #ffc107;padding:4px 8px;border-radius:4px;font-size:12px;color:#856404;margin-top:4px}
.step-data{font-size:11px;color:#999;margin-top:3px;font-family:monospace}
.step-ok .step-num{background:#d4edda;color:#155724}
.step-err .step-num{background:#f8d7da;color:#721c24}
.error-box{background:#f8d7da;border-radius:4px;padding:8px 12px;color:#721c24;font-size:13px}
#progress{height:4px;background:#e9ecef;position:relative;overflow:hidden}
#progress-bar{height:100%;background:#25d366;width:0;transition:width .3s}
</style>
</head>
<body>
<div id="header">
  <h1>🧪 חבילת טסטים אוטומטית — בוט דלתות מיכאל</h1>
  <a href="/test-ui" style="color:#d0ede9;font-size:12px;text-decoration:none">← ממשק צ׳אט</a>
  <button id="btn-run" onclick="runTests()">▶ הרץ את כל הטסטים</button>
</div>
<div id="progress"><div id="progress-bar"></div></div>
<div id="summary">לחץ "הרץ את כל הטסטים" כדי להתחיל</div>
<div id="scenarios"></div>

<script>
const scenariosEl = document.getElementById('scenarios');
const summaryEl   = document.getElementById('summary');
const progressBar = document.getElementById('progress-bar');
const btnRun      = document.getElementById('btn-run');

let cards = {};

function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function initCard(id, name, index, total){
  const card = document.createElement('div');
  card.className = 'card';
  card.id = 'card_' + id;
  card.innerHTML = `
    <div class="card-header" onclick="toggleBody('${id}')">
      <div class="status-dot dot-pend" id="dot_${id}"></div>
      <h2>${esc(name)}</h2>
      <span class="dur" id="dur_${id}"></span>
    </div>
    <div class="card-body" id="body_${id}"></div>`;
  scenariosEl.appendChild(card);
  cards[id] = card;
}

function toggleBody(id){
  document.getElementById('body_' + id).classList.toggle('open');
}

function setRunning(id){
  const dot = document.getElementById('dot_' + id);
  dot.className = 'status-dot dot-running';
}

function setResult(result, index, total){
  const dot  = document.getElementById('dot_' + result.id);
  const dur  = document.getElementById('dur_' + result.id);
  const body = document.getElementById('body_' + result.id);

  dot.className = 'status-dot ' + (result.passed ? 'dot-pass' : 'dot-fail');
  dur.textContent = result.duration_ms + 'ms';
  progressBar.style.width = ((index + 1) / total * 100) + '%';

  if(result.error){
    body.innerHTML = `<div class="error-box">שגיאה: ${esc(result.error)}</div>`;
    body.classList.add('open');
    return;
  }

  let html = '';
  for(const s of result.steps){
    const cls = s.passed ? 'step-ok' : 'step-err';
    const data = Object.entries(s.data)
      .filter(([,v]) => v != null && v !== false && v !== '')
      .map(([k,v]) => `${k}=${JSON.stringify(v)}`)
      .join(' · ');
    const failures = s.failures.map(f => `<div class="step-fail">⚠ ${esc(f)}</div>`).join('');
    html += `<div class="step ${cls}">
      <div class="step-num">${s.turn}</div>
      <div class="step-content">
        <div class="step-user">👤 ${esc(s.message)}</div>
        <div class="step-bot">🤖 ${esc(s.reply.slice(0,200))}${s.reply.length>200?'…':''}</div>
        ${data ? `<div class="step-data">${esc(data)}</div>` : ''}
        ${failures}
      </div>
    </div>`;
  }
  body.innerHTML = html;
  if(!result.passed) body.classList.add('open');
}

async function runTests(){
  btnRun.disabled = true;
  scenariosEl.innerHTML = '';
  summaryEl.innerHTML = '<span class="sum-chip chip-pend">טוען רשימת טסטים...</span>';
  progressBar.style.width = '0';
  progressBar.style.background = '#25d366';
  cards = {};

  const list = await fetch('/run-tests/list').then(r=>r.json());
  const total = list.length;
  let passedCount = 0;

  // Create all cards upfront
  list.forEach((s, i) => initCard(s.id, s.name, i, total));

  // Run each scenario one by one
  for(let i = 0; i < list.length; i++){
    const s = list[i];
    setRunning(s.id);
    summaryEl.innerHTML = `<span class="sum-chip chip-pend">🏃 ${i+1} / ${total} — ${esc(s.name)}</span>`;

    let result;
    try{
      const ctrl = new AbortController();
      const timer = setTimeout(()=>ctrl.abort(), 200000);
      const r = await fetch('/run-tests/scenario/' + s.id, {method:'POST', signal:ctrl.signal});
      clearTimeout(timer);
      result = await r.json();
    } catch(e){
      result = {id:s.id, name:s.name, passed:false, error:'timeout/network: '+e.message, steps:[], duration_ms:0};
    }

    if(result.passed) passedCount++;
    setResult(result, i, total);
  }

  const failed = total - passedCount;
  summaryEl.innerHTML =
    `<span class="sum-chip chip-pass">✅ עברו: ${passedCount}</span>` +
    (failed ? `<span class="sum-chip chip-fail">❌ נכשלו: ${failed}</span>` : '') +
    `<span style="color:#888;font-size:12px">מתוך ${total} טסטים</span>`;
  progressBar.style.background = failed ? '#dc3545' : '#28a745';
  btnRun.disabled = false;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 50)
    print("Test server starting on http://localhost:3001/test-ui")
    print("Uses the SAME get_reply as production (Python)")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=3001, log_level="warning")
