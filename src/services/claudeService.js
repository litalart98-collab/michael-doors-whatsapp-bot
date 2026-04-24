const Anthropic = require('@anthropic-ai/sdk');
const fs   = require('fs');
const path = require('path');
const { getContextBlock } = require('../rules/businessRules');
const scenarioService     = require('./scenarioService');
const logger              = require('../utils/logger');

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const PROMPT_PATH = path.join(__dirname, '../prompts/systemPrompt.txt');
const FAQ_PATH    = path.join(__dirname, '../data/faqBank.json');
const CONV_PATH   = path.join(__dirname, '../../conversations.json');

// ─── FAQ bank — loaded once at startup ───────────────────────────────────────
const faqBank = JSON.parse(fs.readFileSync(FAQ_PATH, 'utf-8'));

// ─── Conversation history — persisted to disk ─────────────────────────────────
function loadConversations() {
  try {
    if (fs.existsSync(CONV_PATH)) return JSON.parse(fs.readFileSync(CONV_PATH, 'utf-8'));
  } catch {}
  return {};
}

function saveConversations(convs) {
  try { fs.writeFileSync(CONV_PATH, JSON.stringify(convs)); } catch {}
}

const conversations = loadConversations();

// ─── Israeli time-based greeting ─────────────────────────────────────────────
function israeliGreeting() {
  const hour = Number(
    new Intl.DateTimeFormat('en-US', {
      hour: 'numeric', hour12: false, timeZone: 'Asia/Jerusalem',
    }).format(new Date())
  );
  if (hour >= 5  && hour < 12) return 'בוקר טוב';
  if (hour >= 12 && hour < 17) return 'צהריים טובים';
  if (hour >= 17 && hour < 21) return 'ערב טוב';
  return 'לילה טוב';
}

const COMPANY_PITCH =
  'אנחנו מציעים דלתות כניסה ופנים באיכות הגבוהה ביותר בשוק — ' +
  'מגוון רחב של דגמים ועיצובים בהתאמה אישית, אחריות מקיפה של מעל שנתיים, ' +
  'ואולם תצוגה מרשים בנתיבות שבו תוכלו להתרשם ולמצוא בדיוק את מה שמתאים לבית שלכם. 🚪✨';

// ─── System prompt — reloaded on every request so live edits take effect ─────
function loadSystemPrompt() {
  return fs.readFileSync(PROMPT_PATH, 'utf-8');
}

// ─── FAQ lookup — up to 3 relevant entries per request ───────────────────────
function findRelevantFaqs(userMessage) {
  const msg = userMessage.toLowerCase();
  return faqBank
    .filter(entry => entry.keywords.some(kw => msg.includes(kw.toLowerCase())))
    .slice(0, 3);
}

function formatFaqBlock(faqs) {
  if (faqs.length === 0) return null;
  const lines = faqs.map(f => `[${f.category}] ${f.answer}`);
  return '## מידע רלוונטי מבסיס הידע (לשימוש כהפניה בלבד — אל תעתיק את הניסוח)\n' + lines.join('\n');
}

// ─── Build full system instruction ───────────────────────────────────────────
function buildSystemInstruction(userMessage) {
  const greeting = israeliGreeting();
  const parts = [
    loadSystemPrompt(),
    `## Business context\n${getContextBlock()}`,
    `## Current time context\nGreeting to use: «${greeting}»\n` +
    `CRITICAL: If there is NO prior assistant message in the conversation history — ` +
    `this is the first reply. You MUST embed the greeting inside the opening line, ` +
    `like this: 'היי, תודה שפניתם לדלתות מיכאל, ${greeting} 😊'. ` +
    `Never skip this on a first reply. Never repeat it after the first reply.`,
  ];

  const matched = findRelevantFaqs(userMessage);
  const faqBlock = formatFaqBlock(matched);
  if (faqBlock) parts.push(faqBlock);
  if (matched.length > 0) logger.info(`FAQ match: ${matched.map(f => f.id).join(', ')}`);

  return parts.join('\n\n');
}

// ─── Main entry point ─────────────────────────────────────────────────────────
async function getReply(sender, userMessage, mock = false) {
  if (!conversations[sender]) conversations[sender] = [];

  conversations[sender].push({ role: 'user', content: userMessage });
  if (conversations[sender].length > 40) {
    conversations[sender] = conversations[sender].slice(-40);
  }

  // ── Scenario classifier — only on first message ───────────────────────────
  const isFirstMessage = conversations[sender].length === 1;
  if (isFirstMessage) {
    const scenario = scenarioService.detect(userMessage);
    if (scenario) {
      logger.info(`Scenario: ${scenario.id} | ${sender}`);
      const greeting = israeliGreeting();

      // Pulse 1: greeting + pitch
      const pulse1 = `היי, תודה שפניתם לדלתות מיכאל, ${greeting} 😊\n${COMPANY_PITCH}`;

      // Pulse 2: actual scripted response (strip the duplicate opening line)
      const pulse2 = scenario.response
        .replace(/^היי,?\s*תודה שפניתם לדלתות מיכאל[^\n]*\n?/i, '')
        .trim();

      // Store combined so Claude sees clean history on next turn
      conversations[sender].push({ role: 'assistant', content: pulse1 + '\n\n' + pulse2 });
      saveConversations(conversations);

      return {
        reply_text:              pulse1,
        reply_text_2:            pulse2 || null,
        handoff_to_human:        scenario.handoff_to_human,
        summary:                 scenario.summary,
        preferred_contact_hours: null,
        needs_frame_removal:     scenario.needs_frame_removal,
        needs_installation:      null,
        full_name:               null,
        phone:                   null,
        city:                    null,
        service_type:            null,
      };
    }
  }

  // ── Mock mode — skip Claude entirely ──────────────────────────────────────
  if (mock) {
    const turn = conversations[sender].length;
    const mockReply = `🤖 [מוק סיבוב ${turn}] קלוד היה עונה כאן על: "${userMessage.slice(0, 40)}"`;
    conversations[sender].push({ role: 'assistant', content: mockReply });
    saveConversations(conversations);
    return {
      reply_text: mockReply, reply_text_2: null,
      handoff_to_human: false,
      summary: `Mock mode — turn ${turn}`,
      preferred_contact_hours: null, needs_frame_removal: null,
      needs_installation: null, full_name: null, phone: null,
      city: null, service_type: null,
    };
  }

  // ── Claude ────────────────────────────────────────────────────────────────
  let rawText;
  try {
    logger.info(`Claude request | sender=${sender} | turns=${conversations[sender].length}`);
    const response = await client.messages.create({
      model:      'claude-sonnet-4-6',
      max_tokens: 900,
      system:     buildSystemInstruction(userMessage),
      messages:   conversations[sender],
    }, { timeout: 55_000 });
    rawText = response.content[0].text;
    logger.info(`Claude OK | sender=${sender} | tokens_out=${response.usage?.output_tokens}`);
  } catch (err) {
    logger.error(`Claude API error | sender=${sender} | ${err.message}`);
    const fallback = 'רגע, בודקת 😊 תכתוב לי שוב בעוד רגע ואענה לך';
    conversations[sender].push({ role: 'assistant', content: fallback });
    saveConversations(conversations);
    return {
      reply_text: fallback, reply_text_2: null,
      handoff_to_human: false,
      summary: 'Claude API error — fallback sent',
      preferred_contact_hours: null, needs_frame_removal: null,
      needs_installation: null, full_name: null, phone: null,
      city: null, service_type: null,
    };
  }

  const structured = parseStructuredResponse(rawText, sender);
  logger.info(`Reply OK | sender=${sender} | text=${structured.reply_text.slice(0, 60)}`);

  // Store combined turn so history stays clean
  const historyContent = structured.reply_text +
    (structured.reply_text_2 ? '\n\n' + structured.reply_text_2 : '');
  conversations[sender].push({ role: 'assistant', content: historyContent });
  saveConversations(conversations);

  return structured;
}

// ─── Parse Claude's JSON response ────────────────────────────────────────────
function parseStructuredResponse(rawText, sender) {
  try {
    const cleaned = rawText
      .trim()
      .replace(/^```(?:json)?\s*/i, '')
      .replace(/\s*```$/, '')
      .replace(/^json\s*/i, '')
      .trim();
    // Find first { in case Claude prepended plain text
    const brace = cleaned.indexOf('{');
    const json = brace > 0 ? cleaned.slice(brace) : cleaned;
    const p = JSON.parse(json);

    const reply2Raw = p.reply_text_2;
    const reply2 = reply2Raw && String(reply2Raw).trim() ? String(reply2Raw).trim() : null;

    return {
      reply_text:              String(p.reply_text || '').trim() || 'רגע, בודקת 😊 תכתוב לי שוב',
      reply_text_2:            reply2,
      handoff_to_human:        Boolean(p.handoff_to_human),
      summary:                 String(p.summary || ''),
      preferred_contact_hours: p.preferred_contact_hours ?? null,
      needs_frame_removal:     p.needs_frame_removal ?? null,
      needs_installation:      p.needs_installation ?? null,
      full_name:               p.full_name ?? null,
      phone:                   p.phone ?? null,
      city:                    p.city ?? null,
      service_type:            p.service_type ?? null,
    };
  } catch {
    logger.warn(`Non-JSON response | sender=${sender} | raw: ${rawText.slice(0, 100)}`);
    return {
      reply_text: rawText.trim() || 'רגע, בודקת 😊 תכתוב לי שוב',
      reply_text_2: null,
      handoff_to_human: false,
      summary: 'Parse error — raw reply returned',
      preferred_contact_hours: null, needs_frame_removal: null,
      needs_installation: null, full_name: null, phone: null,
      city: null, service_type: null,
    };
  }
}

// ─── Clear conversation history for a sender ─────────────────────────────────
function clearConversation(sender) {
  delete conversations[sender];
  saveConversations(conversations);
  logger.info(`Conversation cleared | ${sender}`);
}

module.exports = { getReply, clearConversation };
