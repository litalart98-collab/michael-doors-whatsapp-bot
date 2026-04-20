const fs = require('fs');
const path = require('path');
const logger = require('../utils/logger');

const LEADS_FILE      = path.join(__dirname, '../../leads.json');
const TEST_LEADS_FILE = path.join(__dirname, '../../leads_test.json');

function resolveFile(isTest) {
  return isTest ? TEST_LEADS_FILE : LEADS_FILE;
}

function loadLeads(isTest = false) {
  const file = resolveFile(isTest);
  if (!fs.existsSync(file)) return {};
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8'));
  } catch {
    return {};
  }
}

function saveLeads(leads, isTest = false) {
  fs.writeFileSync(resolveFile(isTest), JSON.stringify(leads, null, 2));
}

// result is the structured object returned by claudeService.getReply()
// isTest = true writes to leads_test.json instead of leads.json
function saveLead(sender, userMessage, result = {}, isTest = false) {
  const leads = loadLeads(isTest);

  if (!leads[sender]) {
    leads[sender] = {
      phone: sender,
      firstContact: new Date().toISOString(),
      isTest: isTest || undefined,
      messages: [],
    };
    logger.info(`${isTest ? 'TEST ' : ''}New lead: ${sender}`);
  }

  const lead = leads[sender];
  lead.lastMessage = new Date().toISOString();

  // Update structured fields if Claude returned them
  if (result.preferred_contact_hours) {
    lead.preferred_contact_hours = result.preferred_contact_hours;
  }
  if (result.needs_frame_removal !== null && result.needs_frame_removal !== undefined) {
    lead.needs_frame_removal = result.needs_frame_removal;
  }
  if (result.needs_installation !== null && result.needs_installation !== undefined) {
    lead.needs_installation = result.needs_installation;
  }
  if (result.handoff_to_human) {
    lead.handoff_to_human = true;
    lead.handoff_time = new Date().toISOString();
  }
  if (result.summary) {
    lead.summary = result.summary;
  }

  lead.messages.push({
    from: 'customer',
    text: userMessage,
    time: new Date().toISOString(),
  });

  if (result.reply_text) {
    lead.messages.push({
      from: 'bot',
      text: result.reply_text,
      time: new Date().toISOString(),
    });
  }

  saveLeads(leads, isTest);
}

function getAllLeads(isTest = false) {
  return loadLeads(isTest);
}

module.exports = { saveLead, getAllLeads };
