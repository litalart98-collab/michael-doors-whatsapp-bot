const axios = require('axios');
const logger = require('../utils/logger');

const INSTANCE_ID = process.env.GREEN_API_INSTANCE_ID;
const TOKEN       = process.env.GREEN_API_TOKEN;
const BASE_URL    = `https://api.green-api.com/waInstance${INSTANCE_ID}`;

// Sends a WhatsApp message. Retries once automatically on failure.
// Timeout: 15 seconds per attempt.
async function sendMessage(chatId, message) {
  const url     = `${BASE_URL}/sendMessage/${TOKEN}`;
  const payload = { chatId, message };

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const response = await axios.post(url, payload, { timeout: 15_000 });
      logger.info(`Green-API sendMessage OK | chatId=${chatId} | attempt=${attempt}`);
      return response.data;
    } catch (err) {
      logger.error(`Green-API sendMessage failed | attempt=${attempt} | chatId=${chatId} | ${err.message}`);
      if (attempt === 2) throw err; // give up after 2 attempts
      await new Promise(r => setTimeout(r, 2000)); // wait 2s before retry
    }
  }
}

module.exports = { sendMessage };
