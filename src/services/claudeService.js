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

// ─── System prompt — reloaded on every request so edits take effect live ─────
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
  return (
    `## מידע רלוונטי מבסיס הידע (לשימוש כהפניה בלבד — אל תעתיק את הניסוח)\n` +
    lines.join('\n')
  );
}

// ─── Build full system instruction ───────────────────────────────────────────
function buildSystemInstruction(userMessage) {
  const parts = [
    loadSystemPrompt(),
    `## Business context\n${getContextBlock()}`,
  ];

  const matched = findRelevantFaqs(userMessage);
  const faqBlock = formatFaqBlock(matched);
  if (faqBlock) parts.push(faqBlock);

  if (matched.length > 0) {
    logger.info(`FAQ match: ${matched.map(f => f.id).join(', ')}`);
  }

  return parts.join('\n\n');
}

// ─── Main entry point ─────────────────────────────────────────────────────────
async function getReply(sender, userMessage) {
  // 1. Init conversation if needed
  if (!conversations[sender]) conversations[sender] = [];

  // 2. Add user message
  conversations[sender].push({ role: 'user', content: userMessage });

  // Keep last 20 turns
  if (conversations[sender].length > 20) {
    conversations[sender] = conversations[sender].slice(-20);
  }

  // 3. Scenario classifier — only on the very first message
  const isFirstMessage = conversations[sender].length === 1;
  if (isFirstMessage) {
    const scenario = scenarioService.detect(userMessage);
    if (scenario) {
      logger.info(`Scenario: ${scenario.id} | ${sender}`);
      conversations[sender].push({ role: 'assistant', content: scenario.response });
      saveConversations(conversations);
      return {
        reply_text:              scenario.response,
        handoff_to_human:        scenario.handoff_to_human,
        summary:                 scenario.summary,
        preferred_contact_hours: null,
        needs_frame_removal:     scenario.needs_frame_removal,
        needs_installation:      null,
      };
    }
  }

  // 4. Claude — 55-second timeout
  let rawText;
  try {
    logger.info(`Claude request | sender=${sender} | turns=${conversations[sender].length}`);
    const response = await client.messages.create({
      model:      'claude-sonnet-4-6',
      max_tokens: 600,
      system:     buildSystemInstruction(userMessage),
      messages:   conversations[sender],
    }, { timeout: 55_000 });
    rawText = response.content[0].text;
  } catch (err) {
    logger.error(`Claude API error | sender=${sender} | ${err.message}`);
    const fallback = 'מצטערים, אירעה תקלה זמנית. אנא נסו שנית בעוד רגע.';
    conversations[sender].push({ role: 'assistant', content: fallback });
    saveConversations(conversations);
    return {
      reply_text:              fallback,
      handoff_to_human:        false,
      summary:                 'Claude API error — fallback sent',
      preferred_contact_hours: null,
      needs_frame_removal:     null,
      needs_installation:      null,
    };
  }

  // 5. Parse response
  const structured = parseStructuredResponse(rawText, sender);
  logger.info(`Reply created | sender=${sender} | text=${structured.reply_text}`);

  // 6. Save to history
  conversations[sender].push({ role: 'assistant', content: structured.reply_text });
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
      .trim();
    const parsed = JSON.parse(cleaned);
    return {
      reply_text:              String(parsed.reply_text || ''),
      handoff_to_human:        Boolean(parsed.handoff_to_human),
      summary:                 String(parsed.summary || ''),
      preferred_contact_hours: parsed.preferred_contact_hours ?? null,
      needs_frame_removal:     parsed.needs_frame_removal ?? null,
      needs_installation:      parsed.needs_installation ?? null,
    };
  } catch {
    logger.warn(`Non-JSON response | sender=${sender} — using raw text`);
    return {
      reply_text:              rawText,
      handoff_to_human:        false,
      summary:                 'Parse error — raw reply returned',
      preferred_contact_hours: null,
      needs_frame_removal:     null,
      needs_installation:      null,
    };
  }
}

// ─── Clear conversation history for a sender ─────────────────────────────────
// Called on every new #test session so Claude starts fresh.
function clearConversation(sender) {
  if (conversations[sender]) {
    delete conversations[sender];
    saveConversations(conversations);
    logger.info(`Conversation cleared | ${sender}`);
  }
}

module.exports = { getReply, clearConversation };
