function timestamp() {
  return new Date().toISOString();
}

const logger = {
  info: (msg) => console.log(`[${timestamp()}] INFO: ${msg}`),
  error: (msg) => console.error(`[${timestamp()}] ERROR: ${msg}`),
  warn: (msg) => console.warn(`[${timestamp()}] WARN: ${msg}`),
};

module.exports = logger;
