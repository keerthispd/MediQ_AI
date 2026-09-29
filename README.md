# AI Medical Assistant Bot

An AI-powered Medical Assistant built using **React**, **FastAPI**, **SQLite**, and a **local language model** served by [Ollama](https://ollama.com). The application provides educational medical guidance, symptom triage, conversation history, safety filtering, red-flag detection, and medical report analysis.

Everything runs on your own machine by default: no API keys, and conversations and reports never leave it. An [optional hosted fallback](#optional-hosted-fallback) can be configured for when the local model is unavailable — it is never used without asking you first.

---

## Features

* Medical question answering with local language models (Qwen2.5 1.5B Instruct for chat, Qwen3 4B Instruct for reports and emergencies)
* Replies stream in as they are written, with follow-up questions understood in context
* Medical report analysis for PDFs, text files, photos and scanned documents (read by a local vision model)
* Symptom triage and educational guidance
* Medical emergency red-flag detection, answered with fixed first-aid steps that appear in milliseconds, before the model has written a word
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
* `qwen2.5:1.5b-instruct` for everyday chat
* `qwen3:4b-instruct` for report analysis and possible emergencies
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
2. Download the models (about 5.5 GB in total):

```bash
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen3:4b-instruct
ollama pull qwen3-vl:2b-instruct
ollama pull all-minilm            # only for grounded answers; see docs/RAG.md
```

A computer with 16 GB of RAM can run these on the CPU; a GPU makes replies much faster.

Chat uses two models, because on a CPU reply time scales with model size. Questions go to the
small `OLLAMA_FAST_MODEL`, which generates about twice as fast; uploaded reports go to the larger
`OLLAMA_MODEL`, where the better reasoning is worth the extra seconds and nobody is watching a
clock. Set either in `.env` (see `.env.example`), along with `OLLAMA_VISION_MODEL`. Prefer
*instruct* models: "thinking" models spend a long time reasoning before they answer.

Possible emergencies also go to `OLLAMA_MODEL`, and both models are loaded at startup, so no
message waits for weights and nothing has to be swapped in. That costs about 5.3 GB of resident
memory; `OLLAMA_EMERGENCY_MODEL=qwen2.5:1.5b-instruct` puts everything on one model on a machine
that cannot spare it. [Emergency messages](#emergency-messages) explains the trade.

### Optional hosted fallback

If the local model cannot run — Ollama isn't installed, the machine is too small, or a model is
still downloading — the app can fall back to a hosted API instead. It supports the **Anthropic
Claude** API (`pip install anthropic`) and any **OpenAI-compatible** endpoint, which covers OpenAI,
Groq, OpenRouter, Together, vLLM and LM Studio.

**This is off unless you configure it, and it is never used silently.** Configuring a provider does
not switch the app over. When the local model fails, the chat explains what happened and asks
whether to continue; only if you accept is your message — and any report you uploaded — sent to
that provider over the internet, where it is covered by their privacy policy rather than staying on
your machine. Replies that came from the hosted model are labelled in the conversation, and you can
switch it back off at any time from the chat header.

Set `CLOUD_PROVIDER` plus the matching key in `.env`; see `.env.example` for every option.

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
* Emergency red-flag detection: a warning, and the first-aid steps for that symptom, are always
  shown before the answer
* A second tier for lone symptoms that are usually ordinary: possible causes with a suggestion
  on each, a question or two, and a fixed line on when it would stop being ordinary - no alarm
* Educational-use disclaimer
* No direct diagnosis functionality
* Uploaded reports are analyzed in memory and never stored

### Emergency messages

An emergency reply has to be fast *and* right, and on a CPU-only machine the local model can
promise neither: it needs tens of seconds to reach its first paragraph, and what it writes is
whatever the model decides to say. So the part that matters does not come from the model.
Red flags are matched by phrase in well under a millisecond, and the warning plus the steps for
that symptom - call your emergency number, what to do while you wait - are fixed text in
`backend/services/redflag.py`. They are streamed before retrieval runs and before the model is
asked anything, so they land in a few milliseconds and are still there when Ollama is down.

Detection matches how people actually write - "my face is drooping", "I can't lift my arm",
"she passed out and I can't wake her" - across collapse, choking, anaphylaxis, heavy bleeding,
breathing trouble, seizures, cardiac and stroke symptoms, bleeding in pregnancy, and poisoning.
It is deliberately biased toward false positives: "what are the symptoms of a heart attack"
trips it, because a caution box nobody needed costs a reader nothing next to a missed heart
attack. `tests/test_emergency.py` pins twenty real phrasings that must be caught and ten
ordinary messages that must not.

The model then adds background on a short leash: `OLLAMA_MODEL`, warmed *and* prompt-primed at
startup, no retrieval (its passages would be read for seconds to add context nobody needs while
calling an ambulance), and a 120-token cap. It is asked only for what the symptoms can mean and
what the medical team will do, and told to give no instructions at all - not even
harmless-sounding ones. Asked merely not to *contradict* the steps, qwen2.5:1.5b ended a
chest-pain reply with "Stay calm, lie down", directly against the steps above it. The reply
therefore also **ends** in fixed text, saying the steps and the call handler come first.

The larger model writes that paragraph on purpose. Over six samples of three emergencies the 1.5B
model produced "they may perform a scope to check the heart and lungs" for chest pain; the 4B
model named ECG, troponin, thrombolytics and angioplasty, correctly, every time. Background that
is wrong is worse than background that is slow, precisely because the steps above it are already
doing the urgent work.

Measured on an i7-1355U, CPU only, with chat and emergencies alternating as they do in use:

| | Warning and steps on screen | Whole reply |
| --- | --- | --- |
| Before (4B, retrieved passages, 400 tokens, cold) | 0.02 s, one sentence of it | 134.6 s |
| **Now** (fixed steps, primed 4B, no retrieval, 120 tokens) | **0.05 s, all of it** | **24-31 s** |
| Same, with `OLLAMA_EMERGENCY_MODEL` set to the 1.5B model | 0.05 s, all of it | ~7 s |

The first column is the one that matters and it does not move, because nothing in it waits for a
model at all: the seconds in the second column are a background paragraph arriving underneath
something the person can already act on. `tests/test_emergency.py` holds that line - the steps
have to survive Ollama being down, and retrieval has to stay uncalled.

### When it is not an emergency

Most symptoms have several causes and the common ones are common. "I have trouble breathing" is
a blocked nose, a cold or anxiety far more often than it is a heart attack, so answering it with
a red warning is wrong most of the time and frightening every time. Phrases like that raise a
**caution** instead of an emergency (`AMBIGUOUS_PHRASES` in `redflag.py`), and a caution is
answered, not alarmed:

- the possible causes, commonest and least serious first, **each with a suggestion attached** -
  something to try for the everyday ones, "worth getting checked by a doctor" for the rest
- one or two questions that would tell those causes apart
- one fixed line, from `SAFETY_NET`, on exactly what would make it urgent

The alarm is one qualifier away. `sudden`, `severe`, `crushing` or `worst` escalates it, so does
a second symptom from another system (chest pain *and* breathlessness), and so does any phrase
specific enough to stand on its own - `gasping`, `blue lips`, `unresponsive`, any stroke sign.
Bleeding in pregnancy never waits.

**Chest pain does both** (`ALARM_AND_ASK`). Most chest pain is not cardiac - reflux, muscle strain
and anxiety are all commoner - but it is the symptom where being wrong the other way costs the
most, so a bare "I have chest pain" gets the full warning and the first-aid steps *and* spends the
model's few words on the two questions that would narrow it down, the ones the call handler asks:
does it spread to your arm or jaw, how long has it lasted. Answer them and the next reply is the
explanation - measured, "hurts more when I press on it" came back as costochondritis, with what to
do about it and how long it takes to settle.

**Answering the questions ends the loop.** Once the user replies, the reply they get is the
fuller one the questions were asked for - which cause now fits and why, what to do about it, how
long to give it, when to see someone - not the same list and another round of questions. The app
also strips its own fixed paragraphs out of the conversation history before sending it back to
the model: shown its own safety net, the model wrote a second, worse one directly above it.

These answers use `OLLAMA_MODEL` too. Putting causes in order is something the small model
measurably does not do - given the retrieved page it transcribes that page's list in the page's
own order, so chest pain opened with angina rather than muscle strain. `OLLAMA_CAUTION_MODEL`
trades that back for speed.

The steps themselves are ordinary Markdown in `redflag.py` - have a clinician review them, and edit
them to match the emergency numbers and services where your users are.

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
