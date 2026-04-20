// Operational facts injected into every Claude request as context.
// Do NOT add pricing here — pricing questions must always be routed to a human.
// Do NOT duplicate instructions already in systemPrompt.txt.

const RULES = {
  businessName: 'Michael Doors',
  contactPhone: '052-000-0000',
  products: [
    'Interior doors',
    'Exterior doors',
    'Security doors',
    'Sliding doors',
    'Aluminum windows',
    'PVC windows',
    'Shutters',
  ],
  workingHours: {
    start: 8,   // 08:00
    end: 20,    // 20:00
    timezone: 'Asia/Jerusalem',
  },
};

function isWorkingHours() {
  const now = new Date();
  const hour = Number(
    new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      hour12: false,
      timeZone: RULES.workingHours.timezone,
    }).format(now)
  );
  return hour >= RULES.workingHours.start && hour < RULES.workingHours.end;
}

// Returns a short context block appended to the system prompt on every request.
function getContextBlock() {
  const hours = isWorkingHours()
    ? `within working hours (${RULES.workingHours.start}:00–${RULES.workingHours.end}:00)`
    : `outside working hours — let the customer know and offer to schedule a callback`;

  return [
    `Business: ${RULES.businessName}`,
    `Phone: ${RULES.contactPhone}`,
    `Products: ${RULES.products.join(', ')}`,
    `Current time status: ${hours}`,
  ].join('\n');
}

module.exports = { getContextBlock, isWorkingHours, RULES };
