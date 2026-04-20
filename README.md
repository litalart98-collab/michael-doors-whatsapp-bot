# Michael Doors — WhatsApp AI Bot

A WhatsApp AI sales assistant for a doors and windows business, powered by Green-API and Claude.

## How it works

1. Customer sends a WhatsApp message
2. Green-API delivers it to this server via webhook
3. Claude generates a reply using the system prompt and business rules
4. The reply is sent back to the customer via Green-API
5. Lead info is saved to `leads.json`

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment variables

Edit `.env` and fill in your credentials:

```
GREEN_API_INSTANCE_ID=your_instance_id
GREEN_API_TOKEN=your_api_token
ANTHROPIC_API_KEY=your_claude_api_key
PORT=3000
```

- **Green-API**: Sign up at https://green-api.com, create an instance, and scan the QR code with your WhatsApp number
- **Anthropic API**: Get your key at https://console.anthropic.com

### 3. Expose your server to the internet

Green-API needs a public URL to send webhooks. Use [ngrok](https://ngrok.com) for local development:

```bash
ngrok http 3000
```

Copy the `https://....ngrok.io` URL for the next step.

### 4. Set the webhook URL in Green-API

In your Green-API dashboard, go to your instance settings and set:

```
Webhook URL: https://your-domain.com/webhook
```

### 5. Start the bot

```bash
# Production
npm start

# Development (auto-restart on changes)
npm run dev
```

## Customize

| File | What to edit |
|------|-------------|
| `src/prompts/systemPrompt.txt` | Bot personality, tone, and instructions |
| `src/rules/businessRules.js` | Products, prices, working hours, phone number |
| `src/services/leadsService.js` | Lead storage logic |

## Project Structure

```
src/
  server.js              # Express app entry point
  routes/
    greenWebhook.js      # Handles incoming WhatsApp messages
  services/
    greenApiService.js   # Sends messages via Green-API
    claudeService.js     # Generates replies with Claude AI
    leadsService.js      # Saves lead data to leads.json
  prompts/
    systemPrompt.txt     # AI system prompt
  rules/
    businessRules.js     # Business hours, products, prices
  utils/
    logger.js            # Simple console logger
```
