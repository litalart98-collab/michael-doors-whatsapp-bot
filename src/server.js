require('dotenv').config({ path: require('path').join(__dirname, '../.env'), override: true });
const express = require('express');
const greenWebhook = require('./routes/greenWebhook');
const logger = require('./utils/logger');

const app = express();
app.use(express.json());

app.get('/', (req, res) => res.json({ status: 'Bot is running' }));
app.use('/webhook', greenWebhook);

// ─── Conversations viewer ──────────────────────────────────────────────────────
const fs   = require('fs');
const path = require('path');

app.get('/conversations', (req, res) => {
  const isTest   = req.query.test !== 'false';
  const file     = path.join(__dirname, isTest ? '../leads_test.json' : '../leads.json');
  const leads    = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf-8')) : {};
  const entries  = Object.values(leads);

  if (req.query.format === 'json') {
    return res.json(leads);
  }

  const rows = entries.flatMap(lead =>
    (lead.messages || []).map(m => `
      <tr class="${m.from === 'bot' ? 'bot' : 'customer'}">
        <td>${lead.phone}</td>
        <td>${m.from === 'bot' ? '🤖 בוט' : '👤 לקוח'}</td>
        <td style="white-space:pre-wrap">${m.text.replace(/</g,'&lt;')}</td>
        <td>${new Date(m.time).toLocaleString('he-IL')}</td>
      </tr>`)
  ).join('');

  res.send(`<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
  <meta charset="UTF-8">
  <title>שיחות בוט - דלתות מיכאל</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
    h1 { color: #333; }
    table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
    th { background: #075e54; color: white; padding: 10px 14px; text-align: right; }
    td { padding: 8px 14px; border-bottom: 1px solid #eee; vertical-align: top; max-width: 400px; }
    tr.bot td { background: #dcf8c6; }
    tr.customer td { background: #fff; }
    tr:hover td { opacity: 0.85; }
    .meta { color: #888; font-size: 12px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <h1>שיחות בוט — דלתות מיכאל</h1>
  <p class="meta">${entries.length} לידים | ${entries.flatMap(l => l.messages || []).length} הודעות
    &nbsp;|&nbsp; <a href="/conversations?test=false">פרודקשן</a>
    &nbsp;|&nbsp; <a href="/conversations?test=true">טסט</a>
    &nbsp;|&nbsp; <a href="/conversations?format=json">JSON</a>
  </p>
  <table>
    <thead><tr><th>מספר</th><th>שולח</th><th>הודעה</th><th>זמן</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="4" style="text-align:center;color:#999">אין שיחות עדיין</td></tr>'}</tbody>
  </table>
</body>
</html>`);
});

// ─── Test Chat API ─────────────────────────────────────────────────────────────
const { getReply, clearConversation } = require('./services/claudeService');

app.post('/test-chat', async (req, res) => {
  const { sender = 'test_ui@c.us', message = '', reset = false, mock = false } = req.body || {};
  if (reset) {
    clearConversation(sender);
    return res.json({ ok: true, reset: true });
  }
  if (!message.trim()) return res.status(400).json({ ok: false, error: 'empty message' });
  try {
    const result = await getReply(sender, message, mock);
    res.json({ ok: true, ...result });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ─── Test UI ───────────────────────────────────────────────────────────────────
app.get('/test-ui', (req, res) => {
  res.send(TEST_UI_HTML);
});

const TEST_UI_HTML = `<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>בוט דלתות מיכאל — בדיקות</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#e5ddd5;height:100vh;display:flex;flex-direction:column;overflow:hidden}
#header{background:#075e54;color:#fff;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0;flex-wrap:wrap}
#header h1{font-size:16px;font-weight:600;flex:1;min-width:140px}
.hctrl{display:flex;align-items:center;gap:6px;font-size:13px}
#sender-input{padding:4px 8px;border-radius:4px;border:none;font-size:13px;width:160px;direction:ltr}
#btn-reset{background:#128c7e;color:#fff;border:none;padding:5px 11px;border-radius:4px;cursor:pointer;font-size:13px}
#btn-reset:hover{background:#0d6f63}
#mock-toggle{display:flex;align-items:center;gap:5px;font-size:12px;color:#d0ede9;cursor:pointer}
#chat{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:5px}
.msg-row{display:flex;max-width:78%}
.msg-row.user{align-self:flex-start;flex-direction:row-reverse}
.msg-row.bot{align-self:flex-end}
.bubble{padding:8px 12px;border-radius:8px;font-size:14px;line-height:1.55;white-space:pre-wrap;word-break:break-word;box-shadow:0 1px 2px rgba(0,0,0,.15)}
.user .bubble{background:#dcf8c6;border-bottom-left-radius:2px}
.bot .bubble{background:#fff;border-bottom-right-radius:2px}
.bubble.mock{background:#fff9e6;border-right:3px solid #f0ad4e}
.bubble.scripted{background:#f0fff4;border-right:3px solid #4caf50}
.bubble.handoff{background:#e8f4fd;border-right:3px solid #2196f3}
.msg-time{font-size:10px;color:#999;margin-top:2px;text-align:left}
.msg-badge{font-size:10px;padding:1px 6px;border-radius:10px;margin-bottom:3px;display:inline-block}
.badge-scripted{background:#c8e6c9;color:#1b5e20}
.badge-mock{background:#fff3cd;color:#856404}
.badge-claude{background:#e1f5fe;color:#01579b}
.badge-handoff{background:#bbdefb;color:#0d47a1;font-weight:700}
#meta{background:#f7f7f7;border-top:1px solid #ddd;padding:6px 16px;font-size:11.5px;color:#666;direction:ltr;min-height:26px;font-family:monospace;flex-shrink:0;overflow-x:auto;white-space:nowrap}
#input-area{background:#f0f0f0;padding:8px 12px;display:flex;gap:8px;align-items:flex-end;flex-shrink:0}
#msg-input{flex:1;padding:10px 14px;border-radius:20px;border:none;font-size:14px;font-family:inherit;resize:none;max-height:120px;outline:none;direction:rtl;line-height:1.4}
#btn-send{background:#075e54;color:#fff;border:none;border-radius:50%;width:44px;height:44px;cursor:pointer;font-size:20px;flex-shrink:0;display:flex;align-items:center;justify-content:center}
#btn-send:hover{background:#128c7e}
#btn-send:disabled{background:#aaa;cursor:not-allowed}
.typing-row{align-self:flex-end}
.typing-bubble{background:#fff;border-radius:8px;padding:8px 14px;font-size:13px;color:#888;font-style:italic;box-shadow:0 1px 2px rgba(0,0,0,.1)}
.sys-msg{text-align:center;font-size:11.5px;color:#888;background:rgba(255,255,255,.55);padding:3px 12px;border-radius:10px;align-self:center;margin:2px 0}
</style>
</head>
<body>
<div id="header">
  <h1>🚪 בוט דלתות מיכאל — ממשק בדיקות</h1>
  <label id="mock-toggle" class="hctrl">
    <input type="checkbox" id="mock-cb">
    מוק (ללא קרדיטים)
  </label>
  <div class="hctrl">
    <span>לקוח:</span>
    <input id="sender-input" value="test_1" placeholder="test_1">
  </div>
  <button id="btn-reset">🔄 איפוס שיחה</button>
</div>
<div id="chat"></div>
<div id="meta">מוכן לבדיקה</div>
<div id="input-area">
  <textarea id="msg-input" rows="1" placeholder="כתוב הודעה... (Enter לשליחה)"></textarea>
  <button id="btn-send" title="שלח">➤</button>
</div>

<script>
const chat    = document.getElementById('chat');
const inp     = document.getElementById('msg-input');
const sendBtn = document.getElementById('btn-send');
const metaBar = document.getElementById('meta');
const mockCb  = document.getElementById('mock-cb');

const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const ts  = () => new Date().toLocaleTimeString('he-IL',{hour:'2-digit',minute:'2-digit'});
const sender = () => (document.getElementById('sender-input').value.trim()||'test_1').replace(/[^\\w]/g,'_')+'@c.us';

function scrollDown(){ chat.scrollTop = chat.scrollHeight; }

function sysMsg(txt){
  const d=document.createElement('div');
  d.className='sys-msg'; d.textContent=txt;
  chat.appendChild(d); scrollDown();
}

function addMsg(text, side, type=''){
  const row = document.createElement('div');
  row.className = 'msg-row '+side;
  const badges = {scripted:'📋 תסריט','badge-scripted':true, mock:'🤖 מוק','badge-mock':true, claude:'✨ קלוד','badge-claude':true, handoff:'✅ הועבר לנציג','badge-handoff':true};
  let badge = '';
  if(type==='scripted') badge='<div class="msg-badge badge-scripted">📋 תסריט</div>';
  else if(type==='mock')   badge='<div class="msg-badge badge-mock">🤖 מוק</div>';
  else if(type==='claude') badge='<div class="msg-badge badge-claude">✨ קלוד</div>';
  else if(type==='handoff') badge='<div class="msg-badge badge-handoff">✅ הועבר לנציג</div>';
  row.innerHTML = '<div>'+ badge +'<div class="bubble '+type+'">'+esc(text)+'</div><div class="msg-time">'+ts()+'</div></div>';
  chat.appendChild(row); scrollDown();
  return row;
}

function setMeta(d){
  const p=[];
  if(d.summary) p.push('📝 '+d.summary);
  if(d.service_type) p.push('🚪 '+d.service_type);
  if(d.city) p.push('📍 '+d.city);
  if(d.full_name) p.push('👤 '+d.full_name);
  if(d.phone) p.push('📞 '+d.phone);
  if(d.preferred_contact_hours) p.push('⏰ '+d.preferred_contact_hours);
  if(d.handoff_to_human) p.push('✅ HANDOFF=true');
  metaBar.textContent = p.length ? p.join('  |  ') : '—';
}

async function send(){
  const text = inp.value.trim();
  if(!text || sendBtn.disabled) return;
  inp.value=''; inp.style.height=''; sendBtn.disabled=true;

  addMsg(text,'user');

  const tyRow=document.createElement('div');
  tyRow.className='msg-row bot typing-row';
  tyRow.innerHTML='<div class="typing-bubble">נטלי מקלידה...</div>';
  chat.appendChild(tyRow); scrollDown();

  try{
    const r = await fetch('/test-chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sender:sender(), message:text, mock:mockCb.checked})
    });
    const d = await r.json();
    tyRow.remove();

    if(!d.ok){ metaBar.textContent='שגיאה: '+(d.error||r.status); return; }

    const isMock = d.reply_text && d.reply_text.startsWith('🤖 [מוק');
    const isHandoff = !!d.handoff_to_human;
    const type = isHandoff ? 'handoff' : isMock ? 'mock' : (mockCb.checked ? 'scripted' : 'claude');

    addMsg(d.reply_text,'bot',type);
    if(d.reply_text_2){
      setTimeout(()=>addMsg(d.reply_text_2,'bot', isMock?'mock':'scripted'), 500);
    }
    setMeta(d);
    if(isHandoff) sysMsg('✅ כל הפרטים נאספו — הועבר לנציג');
  }catch(e){
    tyRow.remove(); metaBar.textContent='שגיאת רשת: '+e.message;
  }finally{
    sendBtn.disabled=false; inp.focus();
  }
}

document.getElementById('btn-reset').onclick = async () => {
  await fetch('/test-chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sender:sender(),reset:true})});
  chat.innerHTML=''; metaBar.textContent='שיחה אופסה';
  sysMsg('שיחה חדשה התחילה');
};

sendBtn.onclick = send;
inp.addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();} });
inp.addEventListener('input', ()=>{ inp.style.height='auto'; inp.style.height=Math.min(inp.scrollHeight,120)+'px'; });

sysMsg('ממשק בדיקות — הודעות לא נשלחות לוואטסאפ 🔒');
</script>
</body>
</html>`;

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  logger.info(`Server started on port ${PORT}`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    logger.error(`Port ${PORT} is already in use. Run: pkill -f "node src/server.js" then restart.`);
  } else {
    logger.error(`Server error: ${err.message}`);
  }
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  logger.error(`Uncaught exception: ${err.message}`);
});

process.on('unhandledRejection', (reason) => {
  logger.error(`Unhandled rejection: ${reason}`);
});
