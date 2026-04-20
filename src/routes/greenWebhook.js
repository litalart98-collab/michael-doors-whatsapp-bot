const express        = require('express');
const fs             = require('fs');
const path           = require('path');
const router         = express.Router();
const greenApiService = require('../services/greenApiService');
const claudeService  = require('../services/claudeService');
const leadsService   = require('../services/leadsService');
const logger         = require('../utils/logger');

const TEST_MODE          = process.env.TEST_MODE === 'true';
const TEST_PHONE         = (process.env.TEST_PHONE || '').trim();
const SESSION_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes inactivity

// ─── Session persistence ──────────────────────────────────────────────────────
const SESSION_PATH = path.join(__dirname, '../../sessions.json');

function loadSessions() {
  try {
    if (fs.existsSync(SESSION_PATH)) return JSON.parse(fs.readFileSync(SESSION_PATH, 'utf-8'));
  } catch {}
  return {};
}

function saveSessions(sessions) {
  try { fs.writeFileSync(SESSION_PATH, JSON.stringify(sessions)); } catch {}
}

const activeSessions = loadSessions();

// ─── Session helpers ──────────────────────────────────────────────────────────
function hasActiveSession(sender) {
  if (!activeSessions[sender]) return false;
  const expired = Date.now() - activeSessions[sender] > SESSION_TIMEOUT_MS;
  if (expired) {
    delete activeSessions[sender];
    saveSessions(activeSessions);
    logger.info(`TEST SESSION | Expired (30 min inactivity) | ${sender}`);
    return false;
  }
  return true;
}

function openSession(sender) {
  activeSessions[sender] = Date.now();
  saveSessions(activeSessions);
  claudeService.clearConversation(sender); // fresh start every #test
  logger.info(`TEST SESSION | Opened | ${sender}`);
}

function touchSession(sender) {
  activeSessions[sender] = Date.now();
  saveSessions(activeSessions);
}

function closeSession(sender) {
  delete activeSessions[sender];
  saveSessions(activeSessions);
  logger.info(`TEST SESSION | Closed by #endtest | ${sender}`);
}

// ─── TEST MODE GATE ───────────────────────────────────────────────────────────
// Returns clean message text to process, or null to skip.
function gate(sender, messageText) {
  if (!TEST_MODE) return messageText; // Production: accept all

  const fromTestPhone = sender.includes(TEST_PHONE);
  if (!fromTestPhone) {
    logger.warn(`TEST MODE | Ignored — wrong number | ${sender}`);
    return null;
  }

  const text = messageText.trim();

  // #endtest — close session, no reply
  if (text.toLowerCase() === '#endtest') {
    closeSession(sender);
    return null;
  }

  // #test — open / refresh session, strip prefix
  if (/^#test\b/i.test(text)) {
    openSession(sender);
    const clean = text.replace(/^#test\s*/i, '').trim();
    return clean || null;
  }

  // Follow-up message — must have active session
  if (hasActiveSession(sender)) {
    touchSession(sender);
    logger.info(`TEST SESSION | Follow-up accepted | ${sender}`);
    return text;
  }

  logger.warn(`TEST MODE | Ignored — no active session | Send "#test <message>" to start | ${sender}`);
  return null;
}

// ─── Webhook handler ──────────────────────────────────────────────────────────
router.post('/', async (req, res) => {
  res.sendStatus(200); // respond fast — Green-API requires quick acknowledgement

  const body = req.body;

  const msgType = body?.messageData?.typeMessage;
  const chatId  = body?.senderData?.chatId;
  const sender  = body?.senderData?.sender;

  logger.info(`WEBHOOK | type=${body?.typeWebhook} | msgType=${msgType} | sender=${sender} | chatId=${chatId}`);

  if (body?.typeWebhook !== 'incomingMessageReceived') return;

  // Extract the customer's text regardless of message type.
  // WhatsApp/Green-API stores the text in different fields depending on context:
  //   textMessage       → textMessageData.textMessage
  //   extendedTextMessage → extendedTextMessageData.text
  //   quotedMessage     → textMessageData.textMessage  (user's text)
  //                       OR extendedTextMessageData.text
  // We try all locations and take the first non-empty value.
  const supported = ['textMessage', 'extendedTextMessage', 'quotedMessage'];
  if (!supported.includes(msgType)) {
    logger.warn(`WEBHOOK | Unsupported msgType=${msgType} — skipped`);
    return;
  }

  const rawText =
    body.messageData?.textMessageData?.textMessage ||
    body.messageData?.extendedTextMessageData?.text ||
    null;

  logger.info(`WEBHOOK | msgType=${msgType} | extracted="${rawText}"`);

  if (!chatId || !rawText) {
    logger.warn(`WEBHOOK | Missing chatId=${chatId} or rawText=${rawText} — skipped`);
    return;
  }

  const cleanText = gate(sender, rawText);
  if (!cleanText) return;

  logger.info(`Incoming | sender=${sender} | chatId=${chatId} | text=${cleanText}`);

  try {
    const result = await claudeService.getReply(sender, cleanText);

    const replyText = (result.reply_text || '').trim();
    if (!replyText) {
      logger.warn(`Empty reply_text for ${sender} — skipping send`);
      return;
    }

    logger.info(`Reply generated | ${sender}: ${replyText}`);

    try {
      await greenApiService.sendMessage(chatId, replyText);
    } catch (sendErr) {
      logger.error(`sendMessage ultimately failed | sender=${sender} | chatId=${chatId} | ${sendErr.message}`);
    }

    leadsService.saveLead(sender, cleanText, result, TEST_MODE);

    if (result.handoff_to_human) {
      logger.info(`HANDOFF REQUIRED | ${sender} | ${result.summary}`);
    }

  } catch (err) {
    logger.error(`Error handling message | sender=${sender} | ${err.message}`);
  }
});

module.exports = router;
