# AI Medical Assistant Bot

An AI-powered Medical Assistant built using **React**, **FastAPI**, **SQLite**, and a **local language model** served by [Ollama](https://ollama.com). The application provides educational medical guidance, symptom triage, conversation history, safety filtering, red-flag detection, and medical report analysis. Everything runs on your own machine: no API keys, and conversations and reports never leave it.

---

## Features

* Medical question answering with a local language model (Qwen3 4B Instruct by default)
* Replies stream in as they are written, with follow-up questions understood in context
* Medical report analysis for PDFs, text files, photos and scanned documents (read by a local vision model)
* Symptom triage and educational guidance
* Medical emergency red-flag detection
* Safety filtering with crisis-helpline responses
* User accounts with private conversation history (SQLite)
* Voice dictation, read-aloud and PDF export
* FastAPI backend with REST APIs

---

## Tech Stack

### Frontend

* React
* Axios
* CSS

### Backend

* FastAPI
* SQLAlchemy
* SQLite
* Python
* pypdfium2 and Pillow (document reading)

### AI

* [Ollama](https://ollama.com) running locally
* `qwen3:4b-instruct` for chat and report analysis
* `qwen3-vl:2b-instruct` for reading photos and scanned PDFs

---

## Project Structure

```text
MedicalAssistant_bot/

├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── api.js
│   │   ├── App.jsx
│   │   └── index.js
│   └── package.json
│
├── backend/
│   ├── routes/          # API and auth endpoints
│   ├── models/          # database models
│   ├── services/        # local model, documents, auth, safety, red flags
│   ├── utils/
│   └── main.py
│
├── deploy/nginx/        # web server config for the Docker frontend
├── database/            # SQLite database (created on first run)
├── docs/
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Local Model Setup

1. Install Ollama from https://ollama.com/download and make sure it is running.
2. Download the models (about 4.5 GB in total):

```bash
ollama pull qwen3:4b-instruct
ollama pull qwen3-vl:2b-instruct
```

A computer with 16 GB of RAM can run both on the CPU; a GPU makes replies much faster. To use other models, set `OLLAMA_MODEL` and `OLLAMA_VISION_MODEL` in `.env` (see `.env.example`). Prefer *instruct* models: "thinking" models spend a long time reasoning before they answer.

---

## Backend Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure (optional)

The defaults work with a local Ollama install. To change them, copy `.env.example` to `.env` and edit it.

### 5. Start Backend Server

```bash
uvicorn backend.main:app --reload
```

Backend will run on:

```text
http://localhost:8000
```

---

## Frontend Setup

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start development server:

```bash
npm start
```

Frontend will run on:

```text
http://localhost:3000
```

Create an account on the login screen, then start chatting. The header shows whether the local AI is ready.

---

## Running with Docker

```bash
docker compose up --build
```

This starts Ollama, downloads the models on the first run, and serves the app at http://localhost:3000. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for details.

---

## API Endpoints

All endpoints except health, status and login require an `Authorization: Bearer <token>` header.

### Health and AI Status

```http
GET /api/health
GET /api/status
```

### Accounts

```http
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
```

### Chat with Assistant

```http
POST /api/chat
```

### Analyze Medical Report

```http
POST /api/upload
```

Chat and upload replies are streamed as newline-delimited JSON events (see `backend/routes/api.py`).

### Conversation History

```http
GET    /api/history
DELETE /api/history
```

---

## Running Tests

```bash
pytest -q
```

Tests use a temporary database and a fake model, so Ollama doesn't need to be running.

---

## Safety Features

* Harmful-content filtering
* Self-harm messages get crisis-helpline resources instead of a model reply
* Emergency red-flag detection: a warning is always shown before the answer
* Educational-use disclaimer
* No direct diagnosis functionality
* Uploaded reports are analyzed in memory and never stored

---

## Database

The application stores:

* User accounts (passwords are hashed with scrypt)
* User messages
* Assistant responses
* Red-flag information
* Conversation timestamps

Database engine:

```text
SQLite
```

---

## Disclaimer

This application is intended for educational and informational purposes only. It does not provide professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## Future Enhancements

* Multi-language support
* Cloud deployment
* Mobile application integration

---

## Author

Developed as an AI-powered healthcare assistance project using React, FastAPI, SQLite, and local language models.
