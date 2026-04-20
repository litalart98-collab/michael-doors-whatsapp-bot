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
