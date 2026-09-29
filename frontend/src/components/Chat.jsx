import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { api, errorText, streamForm } from '../api';

const MAX_MESSAGE_CHARS = 4000;
const HISTORY_TURNS = 10;
const STATUS_RETRY_MS = 30000;

const SUGGESTIONS = [
  'I have a headache and mild fever.',
  'What helps a sore throat?',
  'Explain my blood test results.',
];

// Memoised on `text`, so a streaming reply only re-parses the message that is actually growing.
// Without this every token re-parsed the Markdown of every message on screen.
const MarkdownBody = React.memo(function MarkdownBody({ text }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
});

function TypingIndicator({ status }) {
  return (
    <div className="typing-wrap">
      <span className="typing" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span className="typing-label">{status || 'Thinking…'}</span>
    </div>
  );
}

function describeStatus(status) {
  if (!status) return { tone: '', label: 'Checking…', hint: '' };
  if (!status.ollama) {
    return { tone: 'bad', label: 'Local AI offline', hint: 'Ollama is not reachable on this machine.' };
  }
  // The fast model answers everyday questions and emergencies, so a missing one breaks normal chat
  const missing = [...new Set([
    !status.fast_ready && status.fast_model,
    !status.emergency_ready && status.emergency_model,
    !status.model_ready && status.model,
  ].filter(Boolean))];
  if (missing.length) {
    return {
      tone: 'warn',
      label: 'Model missing',
      hint: `Install with: ${missing.map((m) => `ollama pull ${m}`).join(' && ')}`,
    };
  }
  const vision = status.vision_ready ? '' : ` Image reports need: ollama pull ${status.vision_model}`;
  // Emergencies use the chat model unless OLLAMA_EMERGENCY_MODEL says otherwise
  const models = [`Chat: ${status.fast_model}`];
  if (status.emergency_model !== status.fast_model) models.push(`Emergencies: ${status.emergency_model}`);
  models.push(`Reports: ${status.model}`);
  return {
    tone: 'ok',
    label: 'Local AI ready',
    hint: `${models.join('. ')}.${vision}`,
  };
}

/* Shown when the local model could not answer and replying would mean sending the question to a
   hosted service. This is a privacy decision, so it is always the user's to make. */
function ConsentCard({ consent, onAccept, onDecline }) {
  return (
    <div className="consent-card" role="alertdialog" aria-label="Use the hosted AI?">
      <h4>⚠️ Send this question to {consent.label}?</h4>
      <p>
        The local AI could not answer, so nothing has been sent anywhere. Continuing would send{' '}
        <strong>your message, and any report you uploaded, to {consent.label}</strong> over the
        internet, where it is handled under their privacy policy and not by this machine.
      </p>
      <p className="consent-reason">Local AI said: {consent.reason}</p>
      <div className="consent-actions">
        <button type="button" className="btn btn-primary btn-sm" onClick={onAccept}>
          Use {consent.label}
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onDecline}>
          Keep everything local
        </button>
      </div>
    </div>
  );
}

export default function Chat({ messages, setMessages, draft, setDraft }) {
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [aiStatus, setAiStatus] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [atBottom, setAtBottom] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [cloudConsent, setCloudConsent] = useState(false);

  const fileRef = useRef();
  const textRef = useRef(null);
  const streamRef = useRef(null);
  const recognitionRef = useRef(null);
  const utteranceRef = useRef(null);
  const abortRef = useRef(null);
  const pinnedRef = useRef(true);
  const consentRef = useRef(false); // read inside async handlers, where state would be stale
  const lastRequestRef = useRef(null);
  const busy = loading || uploading;

  const toast = useCallback((text, tone = '') => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500);
  }, []);

  // Stay stuck to the newest text unless the user has scrolled up to read something earlier
  const handleScroll = () => {
    const el = streamRef.current;
    if (!el) return;
    const pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    pinnedRef.current = pinned;
    setAtBottom(pinned);
  };

  // Setting scrollTop directly, rather than a smooth scrollIntoView, keeps this cheap enough to run
  // on every streamed token: smooth scrolling queued hundreds of overlapping animations per reply.
  useEffect(() => {
    const el = streamRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const jumpToLatest = () => {
    const el = streamRef.current;
    if (!el) return;
    pinnedRef.current = true;
    setAtBottom(true);
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  };

  // Stop any ongoing speech, dictation or reply if the component unmounts
  useEffect(() => {
    return () => {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      recognitionRef.current?.abort();
      abortRef.current?.abort();
    };
  }, []);

  const refreshStatus = useCallback(() => {
    api.get('/api/status')
      .then((res) => setAiStatus(res.data))
      .catch(() => setAiStatus({ ollama: false }));
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Keep checking while the model is unavailable, e.g. while it is still downloading
  const aiReady = Boolean(
    aiStatus?.ollama && aiStatus?.model_ready && aiStatus?.fast_ready && aiStatus?.emergency_ready,
  );
  const statusKnown = aiStatus !== null;
  useEffect(() => {
    if (!statusKnown || aiReady) return undefined;
    const timer = setInterval(refreshStatus, STATUS_RETRY_MS);
    return () => clearInterval(timer);
  }, [statusKnown, aiReady, refreshStatus]);

  const startListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast('Dictation needs Chrome or Edge.', 'error');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setDraft((prev) => (prev ? prev + ' ' + transcript : transcript));
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    recognition.start();
  };

  /* Send a request and stream the reply into the last assistant message. `buildForm` is kept so the
     same request can be replayed if the user agrees to the hosted model. */
  const runRequest = useCallback(async (url, buildForm, userText, { replaceAssistant = false } = {}) => {
    pinnedRef.current = true;
    setAtBottom(true);
    lastRequestRef.current = { url, buildForm, userText };

    setMessages((m) => {
      const fresh = { role: 'assistant', text: '', pending: true };
      if (replaceAssistant) return [...m.slice(0, -1), fresh];
      return [...m, { role: 'user', text: userText }, fresh];
    });

    const updateReply = (update) => setMessages((m) => {
      const last = m[m.length - 1];
      if (!last?.pending && !last?.consent) return m; // the chat was cleared meanwhile
      return [...m.slice(0, -1), { ...last, ...update(last) }];
    });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamForm(url, buildForm(consentRef.current), (event) => {
        if (event.type === 'meta') updateReply(() => ({ redflag: event.redflag, blocked: event.blocked }));
        else if (event.type === 'status') updateReply(() => ({ status: event.text }));
        else if (event.type === 'delta') updateReply((msg) => ({ text: msg.text + event.text }));
        else if (event.type === 'provider') updateReply(() => ({ viaCloud: true, providerLabel: event.label }));
        else if (event.type === 'sources') updateReply(() => ({ sources: event.sources }));
        else if (event.type === 'consent_required') {
          updateReply(() => ({
            pending: false,
            status: null,
            consent: { reason: event.reason, label: event.label || 'the hosted AI', model: event.model },
          }));
        }
      }, controller.signal);
      updateReply((msg) => (msg.consent ? {} : { pending: false, status: null, text: msg.text || 'No reply.' }));
    } catch (e) {
      if (e.name === 'AbortError') {
        // The user stopped it on purpose: keep whatever had already arrived
        updateReply((msg) => ({ pending: false, status: null, stopped: true, text: msg.text || 'Stopped.' }));
      } else {
        console.error('Request failed', e);
        updateReply(() => ({ pending: false, status: null, error: true, text: errorText(e, e.message || 'Unknown error') }));
      }
    } finally {
      abortRef.current = null;
    }
    refreshStatus();
  }, [setMessages, refreshStatus]);

  async function send() {
    const text = (draft || '').trim();
    if (!text || busy) return;

    // Earlier turns give the assistant context for follow-up questions (failed replies are left out)
    const history = messages
      .filter((m) => !m.error && !m.pending && !m.consent)
      .slice(-HISTORY_TURNS)
      .map(({ role, text: t }) => ({ role, text: t }));

    const buildForm = (allowCloud) => {
      const form = new FormData();
      form.append('message', text);
      form.append('history', JSON.stringify(history));
      if (allowCloud) form.append('allow_cloud', 'true');
      return form;
    };

    setDraft('');
    if (textRef.current) textRef.current.style.height = ''; // back to one line
    setLoading(true);
    await runRequest('/api/chat', buildForm, text);
    setLoading(false);
  }

  const startUpload = useCallback(async (file) => {
    if (!file || busy) return;
    const text = (draft || '').trim();
    const buildForm = (allowCloud) => {
      const form = new FormData();
      form.append('file', file);
      if (text) form.append('message', text);
      if (allowCloud) form.append('allow_cloud', 'true');
      return form;
    };
    setDraft('');
    setUploading(true);
    await runRequest('/api/upload', buildForm, `[Uploaded File: ${file.name}]` + (text ? `\nQuery: ${text}` : ''));
    setUploading(false);
    if (fileRef.current) fileRef.current.value = '';
  }, [busy, draft, runRequest, setDraft]);

  const acceptCloud = async () => {
    const request = lastRequestRef.current;
    if (!request) return;
    consentRef.current = true;
    setCloudConsent(true);
    setLoading(true);
    await runRequest(request.url, request.buildForm, request.userText, { replaceAssistant: true });
    setLoading(false);
  };

  const declineCloud = () => {
    setMessages((m) => {
      const last = m[m.length - 1];
      if (!last?.consent) return m;
      return [...m.slice(0, -1), {
        ...last,
        consent: null,
        text: 'Kept on this machine. Start Ollama, or install the missing model, and try again.',
      }];
    });
  };

  const revokeCloud = () => {
    consentRef.current = false;
    setCloudConsent(false);
    toast('Hosted AI turned off. Replies stay on this machine.');
  };

  const stopReply = () => abortRef.current?.abort();

  const clearChat = () => {
    if (window.confirm('Clear the current chat?')) {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      abortRef.current?.abort();
      setSpeakingIndex(null);
      setMessages([]);
    }
  };

  const handleCopy = async (text, index) => {
    try {
      // navigator.clipboard only exists on HTTPS pages and localhost
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (e) {
      toast('Copying is blocked here — select the text instead.', 'error');
    }
  };

  const handleSpeak = (text, index) => {
    if (!('speechSynthesis' in window)) {
      toast('Text-to-speech is not supported in this browser.', 'error');
      return;
    }
    window.speechSynthesis.cancel();
    if (speakingIndex === index) {
      setSpeakingIndex(null);
      return;
    }
    // Remove common markdown characters so they aren't read out loud
    const cleanText = text.replace(/[#*`_~]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    // cancel() above fires `end` on the previous utterance asynchronously, so only reset
    // the icon if this utterance is still the one playing.
    const done = () => {
      if (utteranceRef.current === utterance) {
        utteranceRef.current = null;
        setSpeakingIndex(null);
      }
    };
    utterance.onend = done;
    utterance.onerror = done;
    utteranceRef.current = utterance;
    setSpeakingIndex(index);
    window.speechSynthesis.speak(utterance);
  };

  const downloadPDF = async () => {
    const element = document.querySelector('.message-stream');
    if (!element) return;
    // Loaded on demand: html2pdf is large and only needed when exporting
    const { default: html2pdf } = await import('html2pdf.js');
    // Export a copy without the scroll limit so the whole conversation is captured
    const copy = element.cloneNode(true);
    copy.style.maxHeight = 'none';
    copy.style.height = 'auto';
    copy.style.overflow = 'visible';
    html2pdf().set({
      margin: 0.5,
      filename: `MediQ_Conversation_${new Date().toISOString().slice(0, 10)}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
    }).from(copy).save();
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) startUpload(file);
  };

  const status = describeStatus(aiStatus);

  return (
    <section className="card chat-card">
      <header className="chat-header">
        <h2>Consultation</h2>
        <span className={`status-chip ${status.tone}`} title={status.hint}>
          <span className="status-dot" />
          {status.label}
        </span>
        {cloudConsent && (
          <button type="button" className="badge badge-cloud" onClick={revokeCloud} title="Stop using the hosted AI">
            ☁️ Hosted AI on — turn off
          </button>
        )}
        <button type="button" className="btn btn-ghost btn-sm" onClick={downloadPDF} disabled={messages.length === 0}>
          Export
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={clearChat} disabled={messages.length === 0}>
          Clear
        </button>
      </header>

      <div className="chat-body">
        <div className="message-stream" ref={streamRef} onScroll={handleScroll} aria-live="polite">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon" aria-hidden="true">🩺</div>
              <h3>How can I help today?</h3>
              <p>
                Describe your symptoms, ask a general health question, or upload a medical report
                for a plain-language summary.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} type="button" className="chip" onClick={() => setDraft(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`message-row ${m.role === 'user' ? 'user' : 'assistant'}${m.error ? ' error' : ''}`}>
              <div className="message-meta">
                <span>{m.role === 'user' ? 'You' : 'MediQ'}</span>
                {m.blocked && <span className="badge badge-danger">Blocked</span>}
                {m.redflag && <span className="badge badge-warning">Red flag</span>}
                {m.stopped && <span className="badge badge-neutral">Stopped</span>}
                {m.viaCloud && <span className="badge badge-cloud">via {m.providerLabel}</span>}
              </div>

              <div className="message-bubble">
                {/* Shown as soon as retrieval finishes, which is well before the first word of the
                    answer: the wait then has something to read. */}
                {m.sources?.length > 0 && (
                  <div className="sources">
                    <span className="sources-label">Based on</span>
                    <ol className="sources-list">
                      {m.sources.map((s, n) => (
                        <li key={s.url}>
                          <a href={s.url} target="_blank" rel="noopener noreferrer">
                            <span className="source-num">{n + 1}</span>{s.title}
                          </a>
                          <span className="source-org">{s.source}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
                {m.pending && !m.text ? <TypingIndicator status={m.status} /> : <MarkdownBody text={m.text} />}
                {m.consent && (
                  <ConsentCard consent={m.consent} onAccept={acceptCloud} onDecline={declineCloud} />
                )}
              </div>

              {m.role === 'assistant' && !m.pending && m.text && (
                <div className="message-actions" data-html2canvas-ignore="true">
                  <button type="button" className="icon-btn" onClick={() => handleSpeak(m.text, i)} aria-label="Read aloud">
                    {speakingIndex === i ? '⏹️' : '🔊'}
                  </button>
                  <button type="button" className="icon-btn" onClick={() => handleCopy(m.text, i)} aria-label="Copy reply">
                    {copiedIndex === i ? '✅' : '📋'}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {!atBottom && messages.length > 0 && (
          <button type="button" className="btn btn-ghost btn-sm jump-latest" onClick={jumpToLatest}>
            ↓ Latest
          </button>
        )}
      </div>

      <div className="composer-panel">
        <div className="composer-row">
          <textarea
            id="message-input"
            ref={textRef}
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              // Grow with the text instead of hiding longer questions behind a one-line scroll
              e.target.style.height = 'auto';
              e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
            }}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter adds a new line (ignored while an IME is composing text)
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Describe your symptoms or ask a health question…"
            maxLength={MAX_MESSAGE_CHARS}
            rows={1}
            aria-label="Your message"
          />
          <button
            type="button"
            className={`btn btn-ghost btn-icon${isListening ? ' is-active' : ''}`}
            onClick={startListening}
            aria-label={isListening ? 'Stop dictation' : 'Dictate your message'}
          >
            {isListening ? '🔴' : '🎤'}
          </button>
          {busy ? (
            <button type="button" className="btn btn-danger" onClick={stopReply}>
              ■ Stop
            </button>
          ) : (
            <button type="button" className="btn btn-primary" onClick={send} disabled={!draft.trim()}>
              Send
            </button>
          )}
        </div>

        <div className="composer-hint">
          <span>{draft.length > MAX_MESSAGE_CHARS - 500 ? `${draft.length} / ${MAX_MESSAGE_CHARS}` : ''}</span>
          <span className="hint-keys">
            <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
          </span>
        </div>

        <div
          className={`upload-panel${dragging ? ' dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <div>
            <strong>Upload a medical report</strong>
            <p>Drag a file here, or choose one. PDF, image or text, up to 20&nbsp;MB. Analysed on this machine and not stored. Remove personal details first.</p>
          </div>
          <label className="btn btn-ghost upload-label">
            {uploading ? 'Analysing…' : 'Choose file'}
            <input
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.webp,.txt"
              onChange={(e) => startUpload(e.target.files[0])}
              ref={fileRef}
              disabled={busy}
              hidden
            />
          </label>
        </div>
      </div>

      {toasts.length > 0 && (
        <div className="toast-stack">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.tone}`} role="status">{t.text}</div>
          ))}
        </div>
      )}
    </section>
  );
}
