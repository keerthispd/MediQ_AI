const { createProxyMiddleware } = require('http-proxy-middleware');

// Development server only: forward API calls to the backend.
// (Used instead of the package.json "proxy" field, which makes `npm start` crash on machines whose
// first network adapter has no private LAN address, e.g. with Tailscale or other VPNs.)
module.exports = function setupProxy(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: process.env.BACKEND_URL || 'http://localhost:8000',
      changeOrigin: true,
    })
  );
};
