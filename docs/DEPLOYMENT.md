# Demo & Deployment

This document covers running a local demo and basic containerized deployment for the AI Medical Assistant project.

Prerequisites

- Docker & Docker Compose (for container deployment)
- Ollama (for local development, https://ollama.com/download)
- Node.js + npm (for local frontend dev)
- Python 3.11+ and a virtualenv (for local backend dev)

Local demo (development)

1. Local models (Ollama)

```bash
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen3:4b-instruct
ollama pull qwen3-vl:2b-instruct
```

2. Backend (Python)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The backend creates the SQLite DB at `database/backend.db` on startup and talks to Ollama at `http://localhost:11434` (change with `OLLAMA_BASE_URL`).

3. Frontend (React)

```bash
cd frontend
npm install
npm start
```

The frontend uses a proxy to forward `/api/*` to `http://localhost:8000` during development.

Containerized demo (recommended for demos)

```bash
docker compose up --build
```

This starts four services:

- `ollama`: the local model server. Models are kept in the `ollama-models` volume.
- `ollama-pull`: a one-off job that downloads the models. The first run downloads about 4.5 GB; later runs finish immediately.
- `backend`: the API on port `8000`.
- `frontend`: Nginx on port `3000`. It serves the built React app and proxies `/api/*` to the backend (config: `deploy/nginx/app.conf`).

The app can be used while the models download; the chat header shows when the AI is ready. To use different models, set `OLLAMA_FAST_MODEL` (everyday chat), `OLLAMA_MODEL` (reports and emergencies) and `OLLAMA_VISION_MODEL` in `.env` before starting. Both images build from the repository root.

The Ollama container runs on the CPU. On a machine with an NVIDIA GPU and the NVIDIA Container Toolkit, give the `ollama` service GPU access (see Ollama's Docker documentation) for much faster replies.

Deploying to production

- Use a production-grade ASGI server (e.g., `uvicorn` with process manager or `gunicorn` + `uvicorn` workers).
- In production, prefer serving the frontend as static files via a CDN or a web server (Nginx). The `frontend/Dockerfile` builds the static files and serves them with Nginx.
- Run Ollama on a machine with a GPU if many people will use the app; CPU generation handles about one reply at a time.
- Consider using managed container services (ECS, GKE, App Service) or a vendor's PaaS and configure environment variables there.

Security & Safety

- The assistant includes a safety layer that answers self-harm messages with crisis resources, and a red-flag detector that shows an emergency warning and fixed first-aid steps before the answer. Review `backend/services/safety.py` and `backend/services/redflag.py` before deploying: both reply to users in wording you ship rather than wording a model chooses, and the emergency numbers and services in them need to match where your users are.
- Users must create an account; each user only sees their own history. Serve the app over HTTPS so passwords and tokens are encrypted in transit.
- Uploaded reports are analyzed in memory and never written to disk.
- Don't expose the Ollama port (`11434`) publicly; it has no authentication.

Monitoring & Reliability

- `GET /api/status` reports whether Ollama is reachable and the models are installed.
- Model requests time out after 300 seconds without output. Replies are capped at about 1,000 tokens (1,500 for reports) so a model can't run forever.
- Add logging and metrics for model latency, errors, and red-flag counts in production.

Further work

- Add infrastructure IaC templates (Terraform) for full deployments.
