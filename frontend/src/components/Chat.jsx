import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { api, errorText, streamForm } from '../api';

const MAX_MESSAGE_CHARS = 4000;
const HISTORY_TURNS = 10;
const STATUS_RETRY_MS = 30000;

const markdownComponents = {
  p: ({node, ...props}) => <p style={{ margin: '0 0 10px 0' }} {...props} />,
  ul: ({node, ...props}) => <ul style={{ margin: '0 0 10px 0', paddingLeft: '20px' }} {...props} />,
  ol: ({node, ...props}) => <ol style={{ margin: '0 0 10px 0', paddingLeft: '20px' }} {...props} />,
  h1: ({node, ...props}) => <h1 style={{ margin: '10px 0', fontSize: '1.2rem' }} {...props} />,
  h2: ({node, ...props}) => <h2 style={{ margin: '10px 0', fontSize: '1.1rem' }} {...props} />,
  h3: ({node, ...props}) => <h3 style={{ margin: '10px 0', fontSize: '1.05rem' }} {...props} />
};

const dotStyle = { width: '8px', height: '8px', backgroundColor: '#94a3b8', borderRadius: '50%', animation: 'bounce 1.4s infinite ease-in-out both' };

function TypingIndicator({ status }) {
  return (
    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', height: '24px' }}>
      <span style={{ ...dotStyle, animationDelay: '-0.32s' }}></span>
      <span style={{ ...dotStyle, animationDelay: '-0.16s' }}></span>
      <span style={dotStyle}></span>
      {status && <span style={{ marginLeft: '6px', fontSize: '0.85rem', color: '#64748b' }}>{status}</span>}
    </div>
  );
}

function describeStatus(status) {
  if (!status) return { label: 'Checking AI…', color: '#cbd5e1', hint: '' };
  if (!status.ollama) return { label: 'AI offline', color: '#fca5a5', hint: 'The local AI server (Ollama) is not reachable.' };
  if (!status.model_ready) return { label: 'AI model missing', color: '#fcd34d', hint: `Install it with: ollama pull ${status.model}` };
  const vision = status.vision_ready ? '' : ` Image reports need: ollama pull ${status.vision_model}`;
  return { label: 'Local AI ready', color: '#86efac', hint: `Model: ${status.model}.${vision}` };
}

export default function Chat({ messages, setMessages, draft, setDraft }) {
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [aiStatus, setAiStatus] = useState(null);
  const fileRef = useRef();
  const textRef = useRef(null);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);
  const utteranceRef = useRef(null);
  const abortRef = useRef(null);
  const busy = loading || uploading;

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Stop any ongoing speech, dictation or reply if the component unmounts
  useEffect(() => {
    return () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
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
  const aiReady = Boolean(aiStatus?.ollama && aiStatus?.model_ready);
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
      alert('Speech recognition is not supported in this browser. Please use Google Chrome or Edge.');
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
    recognition.onerror = (event) => {
      console.error('Speech error', event);
      setIsListening(false);
    };
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    recognition.start();
  };

  // Show the user's message, then stream the backend's reply into a new assistant message
  async function streamReply(url, form, userText) {
    setMessages((m) => [...m, { role: 'user', text: userText }, { role: 'assistant', text: '', pending: true }]);
    const updateReply = (update) => setMessages((m) => {
      const last = m[m.length - 1];
      if (!last?.pending) return m; // the chat was cleared meanwhile
      return [...m.slice(0, -1), { ...last, ...update(last) }];
    });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamForm(url, form, (event) => {
        if (event.type === 'meta') updateReply(() => ({ redflag: event.redflag, blocked: event.blocked }));
        else if (event.type === 'status') updateReply(() => ({ status: event.text }));
        else if (event.type === 'delta') updateReply((msg) => ({ text: msg.text + event.text }));
      }, controller.signal);
      updateReply((msg) => ({ pending: false, status: null, text: msg.text || 'No reply.' }));
    } catch (e) {
      if (e.name !== 'AbortError') console.error('Request failed', e);
      updateReply(() => ({ pending: false, status: null, error: true, text: errorText(e, e.message || 'Unknown error') }));
    } finally {
      abortRef.current = null;
    }
    refreshStatus();
  }

  async function send() {
    const text = (draft || '').trim();
    if (!text || busy) return;

    // Earlier turns give the assistant context for follow-up questions (failed replies are left out)
    const history = messages
      .filter((m) => !m.error && !m.pending)
      .slice(-HISTORY_TURNS)
      .map(({ role, text }) => ({ role, text }));

    const form = new FormData();
    form.append('message', text);
    form.append('history', JSON.stringify(history));
    setDraft('');
    setLoading(true);
    await streamReply('/api/chat', form, text);
    setLoading(false);
  }

  async function uploadFile(e) {
    const file = e.target.files[0];
    if (!file || busy) return;

    // Pass the user's current draft as a query alongside the file upload
    const text = (draft || '').trim();
    const form = new FormData();
    form.append('file', file);
    if (text) form.append('message', text);
    setDraft('');
    setUploading(true);
    await streamReply('/api/upload', form, `[Uploaded File: ${file.name}]` + (text ? `\nQuery: ${text}` : ''));
    setUploading(false);
    if (fileRef.current) fileRef.current.value = '';
  }

  const clearChat = () => {
    if (window.confirm('Are you sure you want to clear the current chat?')) {
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
      alert('Copying is not available here. Please select the text and copy it manually.');
    }
  };

  const handleSpeak = (text, index) => {
    if (!('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported in this browser.');
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
    // Export a copy without the scroll limit so the whole conversation is captured, not just the visible part
    const copy = element.cloneNode(true);
    copy.style.maxHeight = 'none';
    copy.style.height = 'auto';
    copy.style.overflow = 'visible';
    const opt = {
      margin:       0.5,
      filename:     `Medical_Assistant_Report_${new Date().toISOString().slice(0, 10)}.pdf`,
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2 },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(copy).save();
  };

  const status = describeStatus(aiStatus);

  return (
      <section className="chat-card" style={{ backgroundColor: '#ffffff', borderRadius: '16px', boxShadow: '0 10px 40px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div className="chat-header" style={{ padding: '24px', background: 'linear-gradient(135deg, #0369a1 0%, #0f766e 100%)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '1.4rem', color: '#ffffff', fontWeight: '700', letterSpacing: '0.5px' }}>MediQ AI</h2>
          <p style={{ margin: '6px 0 0', fontSize: '0.95rem', color: '#ccfbf1' }}>Empowering your health with smart insights</p>
          <span title={status.hint} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', marginTop: '10px', padding: '3px 10px', borderRadius: '999px', backgroundColor: 'rgba(255,255,255,0.15)', color: '#ffffff', fontSize: '0.8rem', cursor: 'help' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: status.color }}></span>
            {status.label}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={downloadPDF} disabled={messages.length === 0} style={{ padding: '8px 12px', borderRadius: '6px', border: 'none', backgroundColor: 'rgba(255,255,255,0.2)', color: '#ffffff', cursor: messages.length === 0 ? 'not-allowed' : 'pointer', opacity: messages.length === 0 ? 0.6 : 1, fontWeight: '600', fontSize: '0.85rem', transition: 'background 0.2s' }} onMouseEnter={e => e.target.style.backgroundColor='rgba(255,255,255,0.3)'} onMouseLeave={e => e.target.style.backgroundColor='rgba(255,255,255,0.2)'}>📄 Export PDF</button>
          <button onClick={clearChat} style={{ padding: '8px 12px', borderRadius: '6px', border: 'none', backgroundColor: 'rgba(239, 68, 68, 0.8)', color: '#ffffff', cursor: 'pointer', fontWeight: '600', fontSize: '0.85rem', transition: 'background 0.2s' }} onMouseEnter={e => e.target.style.backgroundColor='rgba(239, 68, 68, 1)'} onMouseLeave={e => e.target.style.backgroundColor='rgba(239, 68, 68, 0.8)'}>🗑️ Clear Chat</button>
        </div>
      </div>

      <div className="message-stream" aria-live="polite" style={{ padding: '24px', flex: 1, overflowY: 'auto', minHeight: '450px', backgroundColor: '#f8fafc' }}>
        {messages.length === 0 && (
          <div className="empty-state" style={{ textAlign: 'center', padding: '40px 20px', backgroundColor: '#ffffff', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 15px rgba(0,0,0,0.03)', color: '#0f172a', margin: '40px auto', maxWidth: '80%' }}>
            <div style={{ fontSize: '3rem', marginBottom: '16px' }}>🩺</div>
            <strong style={{ fontSize: '1.15rem', color: '#0369a1', display: 'block', marginBottom: '10px' }}>Welcome to MediQ AI</strong>
            <p style={{ maxWidth: '400px', margin: '0 auto', color: '#64748b', lineHeight: '1.6', fontSize: '0.95rem' }}>
              Describe your symptoms, ask general health questions, or securely upload a medical report for an educational summary.
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`message-row ${m.role === 'user' ? 'user' : 'assistant'}`} style={{ display: 'flex', flexDirection: 'column', alignItems: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: '16px' }}>
            <div className="message-bubble" style={{ background: m.role === 'user' ? 'linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%)' : '#ffffff', color: m.role === 'user' ? '#ffffff' : '#1e293b', padding: '16px 20px', borderRadius: m.role === 'user' ? '16px 16px 0 16px' : '16px 16px 16px 0', maxWidth: '85%', boxShadow: m.role === 'user' ? '0 4px 15px rgba(14, 165, 233, 0.2)' : '0 4px 15px rgba(0,0,0,0.05)', border: m.role === 'assistant' ? '1px solid #e2e8f0' : 'none' }}>
              <div className="message-meta" style={{ fontSize: '0.8rem', marginBottom: '4px', opacity: 0.8, display: 'flex', gap: '8px', alignItems: 'center' }}>
                <strong>{m.role === 'user' ? 'You' : 'Assistant'}</strong>
                {m.blocked && <span className="badge badge-danger" style={{ backgroundColor: '#ef4444', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>Blocked</span>}
                {m.redflag && <span className="badge badge-warning" style={{ backgroundColor: '#f59e0b', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>Red flag</span>}

                {m.role === 'assistant' && !m.pending && (
                  <div data-html2canvas-ignore="true" style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
                    <button onClick={() => handleSpeak(m.text, i)} title="Read aloud" style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1rem', padding: '2px', opacity: 0.7, transition: 'opacity 0.2s' }} onMouseEnter={e => e.target.style.opacity = 1} onMouseLeave={e => e.target.style.opacity = 0.7}>
                      {speakingIndex === i ? '⏹️' : '🔊'}
                    </button>
                    <button onClick={() => handleCopy(m.text, i)} title="Copy to clipboard" style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1rem', padding: '2px', opacity: 0.7, transition: 'opacity 0.2s' }} onMouseEnter={e => e.target.style.opacity = 1} onMouseLeave={e => e.target.style.opacity = 0.7}>
                      {copiedIndex === i ? '✅' : '📋'}
                    </button>
                  </div>
                )}
              </div>
              {m.pending && !m.text ? (
                <TypingIndicator status={m.status} />
              ) : (
                <div className="markdown-body" style={{ margin: 0, lineHeight: '1.6', fontSize: '0.95rem' }}>
                  <ReactMarkdown components={markdownComponents}>{m.text}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="composer-panel" style={{ padding: '20px 24px', borderTop: '1px solid #e2e8f0', backgroundColor: '#ffffff' }}>
        <div className="composer-row" style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <textarea
            id="message-input"
            ref={textRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter adds a new line (ignored while an IME is composing text)
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Describe symptoms, ask about a report, or request next steps..."
            maxLength={MAX_MESSAGE_CHARS}
            rows={1}
            style={{ flex: 1, padding: '10px 12px', borderRadius: '8px', border: '1px solid #cbd5e1', resize: 'none', fontFamily: 'inherit', fontSize: '0.9rem', minHeight: '42px', height: '42px', boxSizing: 'border-box' }}
          />
          <button type="button" onClick={startListening} title={isListening ? 'Stop dictation' : 'Dictate with your voice'} style={{ background: isListening ? '#fee2e2' : '#f8fafc', color: isListening ? '#ef4444' : '#64748b', width: '42px', height: '42px', padding: '0', borderRadius: '8px', border: `1px solid ${isListening ? '#fca5a5' : '#cbd5e1'}`, cursor: 'pointer', transition: 'all 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', boxShadow: isListening ? '0 0 0 4px rgba(239, 68, 68, 0.1)' : 'none', boxSizing: 'border-box' }}>
            {isListening ? '🔴' : '🎤'}
          </button>
          <button className="primary-button" onClick={send} disabled={busy} type="button" style={{ background: 'linear-gradient(135deg, #0ea5e9 0%, #14b8a6 100%)', color: '#fff', height: '42px', padding: '0 20px', borderRadius: '8px', border: 'none', cursor: busy ? 'not-allowed' : 'pointer', fontWeight: '600', transition: 'all 0.2s', opacity: busy ? 0.7 : 1, boxShadow: '0 4px 10px rgba(14,165,233,0.2)', fontSize: '0.9rem', boxSizing: 'border-box' }}>
            {loading ? 'Replying…' : 'Send'}
          </button>
        </div>

        <div className="upload-panel" style={{ marginTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', backgroundColor: '#f0fdfa', border: '1px dashed #5eead4', borderRadius: '12px' }}>
          <div>
            <strong style={{ fontSize: '0.9rem', color: '#0f172a' }}>Upload Medical Report</strong>
            <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b' }}>PDF, image or text file, up to 20 MB. Analyzed privately by the local AI and not stored. Remove PII before upload.</p>
          </div>
          <div className="upload-row" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <label className="upload-label" style={{ cursor: busy ? 'not-allowed' : 'pointer', backgroundColor: '#ffffff', padding: '10px 18px', borderRadius: '8px', fontSize: '0.9rem', fontWeight: '600', color: '#0f766e', border: '1px solid #99f6e4', boxShadow: '0 2px 4px rgba(0,0,0,0.02)', transition: 'all 0.2s' }}>
              {uploading ? 'Analyzing…' : 'Choose File'}
              <input type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,.txt" onChange={uploadFile} ref={fileRef} disabled={busy} style={{ display: 'none' }} />
            </label>
          </div>
        </div>
      </div>
    </section>
  );
}
