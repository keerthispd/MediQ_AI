# Production deployment notes

This file expands on production best practices for deploying the AI Medical Assistant.

CI and Docker images

- A sample CI workflow is included at `.github/workflows/ci.yml`. It installs backend dependencies and runs `pytest`, then builds Docker images for backend and frontend. Images are built locally in the Actions runner and not pushed by default.
- To push images, add registry credentials as repository secrets and update the workflow to use `docker/login-action` and set `push: true` in `docker/build-push-action` steps.

Nginx reverse proxy

- Example config: `deploy/nginx/app.conf`. It serves the frontend static files and proxies `/api/` to the backend with response buffering off, so streamed replies reach the browser as they are generated.
- In production, run Nginx in front of the backend and frontend. Configure TLS (Let's Encrypt or managed certificates) and redirect HTTP to HTTPS.
- Any other proxy in front of the backend must also pass streamed responses through without buffering and allow long read timeouts (several minutes on CPU-only hosts).

Local model server

- The app needs no API keys: models run in Ollama. Keep the Ollama port private to the backend's network.
- Size the host for the models: `qwen2.5:1.5b-instruct` needs about 2 GB of memory, `qwen3:4b-instruct` about 4 GB and `qwen3-vl:2b-instruct` about 3 GB. Chat keeps the fast model loaded and only loads the larger one for reports and emergencies. A GPU greatly reduces reply times.
- Configure the models, context size and keep-alive with the `OLLAMA_*` variables in `.env.example`.

Secrets management

- Keep `.env` out of the repository (it is git-ignored). Use environment variables injected by your deployment platform or a secrets manager for anything sensitive, such as a production `DATABASE_URL`.
- GitHub Actions: store secrets under `Settings → Secrets and variables → Actions` and reference them as `${{ secrets.NAME }}`.

Observability and monitoring

- Add structured logging and export logs to a centralized system (Cloud Logging, ELK, Datadog).
- Export basic metrics: request latency, model latency, model failures, red-flag counts. For
  red-flag messages the number to watch is time to the first streamed event, not time to the last
  one: the first two carry the emergency warning and the first-aid steps.
- Poll `GET /api/status` to alert when Ollama is down or a model is missing.

Security

- Review the safety layer in `backend/services/safety.py` and `backend/services/redflag.py` before
  production. Both contain text sent to users verbatim, including the first-aid steps and the
  emergency numbers shown for a red flag; have a clinician check them for the countries you serve.
- Accounts use scrypt password hashes and expiring bearer tokens (`AUTH_TOKEN_DAYS`). Consider adding rate limiting on `/api/auth/login`.
- Uploaded reports are processed in memory and never stored.
