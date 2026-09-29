import React, { useCallback, useEffect, useState } from 'react';
import Chat from './components/Chat';
import SafetyDisclaimer from './components/SafetyDisclaimer';
import History from './components/History';
import { api, clearSession, errorText, loadSession, saveSession, setUnauthorizedHandler } from './api';

const FEATURES = [
  { icon: '🔒', label: 'Runs on your machine by default' },
  { icon: '📄', label: 'Plain-language report summaries' },
  { icon: '⚠️', label: 'Emergency red-flag detection' },
  { icon: '🎙️', label: 'Dictation and read-aloud' },
];

const THEME_KEY = 'mediq_theme';

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(THEME_KEY) || 'system';
    } catch (e) {
      return 'system';
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (e) {
      // Private browsing blocks storage; the theme still applies for this session
    }
  }, [theme]);

  return [theme, setTheme];
}

function ThemeToggle({ theme, setTheme }) {
  const next = { system: 'light', light: 'dark', dark: 'system' }[theme];
  const icon = { system: '🖥️', light: '☀️', dark: '🌙' }[theme];
  return (
    <button
      type="button"
      className="btn btn-ghost btn-icon"
      onClick={() => setTheme(next)}
      title={`Theme: ${theme}. Switch to ${next}.`}
      aria-label={`Theme: ${theme}. Switch to ${next}.`}
    >
      {icon}
    </button>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [user, setUser] = useState(() => loadSession().username || '');
  const [authMode, setAuthMode] = useState('login');
  const [authError, setAuthError] = useState('');
  const [authBusy, setAuthBusy] = useState(false);
  const [theme, setTheme] = useTheme();

  const endSession = useCallback(() => {
    clearSession();
    setUser('');
    setMessages([]);
    setDraft('');
    setAuthMode('login');
  }, []);

  // Any request rejected as unauthenticated (e.g. an expired session) returns to the login screen
  useEffect(() => {
    setUnauthorizedHandler(endSession);
  }, [endSession]);

  // Check a saved session is still valid when the app opens
  useEffect(() => {
    if (user) api.get('/api/auth/me').catch(() => {});
  }, [user]);

  const handleAuth = async (e) => {
    e.preventDefault();
    const form = e.target;
    setAuthBusy(true);
    setAuthError('');
    try {
      const res = await api.post(`/api/auth/${authMode}`, {
        username: form.username.value.trim(),
        password: form.password.value,
      });
      saveSession(res.data);
      setUser(res.data.username);
    } catch (err) {
      setAuthError(errorText(err, 'Something went wrong. Please try again.'));
    } finally {
      setAuthBusy(false);
    }
  };

  const handleLogout = async () => {
    try {
      await api.post('/api/auth/logout');
    } catch (e) {
      // The session is cleared locally either way
    }
    endSession();
  };

  if (!user) {
    const registering = authMode === 'register';
    return (
      <div className="app-shell auth-shell">
        <div className="card auth-card">
          <div className="brand-mark" aria-hidden="true">🩺</div>
          <h2>MediQ AI</h2>
          <p>Private medical guidance that runs on your own machine.</p>

          <form onSubmit={handleAuth} className="auth-form">
            <input
              name="username"
              type="text"
              placeholder="Username"
              autoComplete="username"
              aria-label="Username"
              required
              minLength={registering ? 3 : undefined}
              maxLength={32}
              pattern={registering ? '[A-Za-z0-9_.\\-]+' : undefined}
              title={registering ? 'Letters, numbers, dots, dashes and underscores only' : undefined}
            />
            <input
              name="password"
              type="password"
              placeholder={registering ? 'Password (at least 8 characters)' : 'Password'}
              autoComplete={registering ? 'new-password' : 'current-password'}
              aria-label="Password"
              required
              minLength={registering ? 8 : undefined}
              maxLength={128}
            />
            {authError && <div role="alert" className="auth-error">{authError}</div>}
            <button type="submit" className="btn btn-primary" disabled={authBusy}>
              {authBusy ? 'Please wait…' : registering ? 'Create account' : 'Log in'}
            </button>
          </form>

          <p className="auth-switch">
            {registering ? 'Already have an account?' : 'New to MediQ?'}{' '}
            <button
              type="button"
              className="link-button"
              onClick={() => { setAuthMode(registering ? 'login' : 'register'); setAuthError(''); }}
            >
              {registering ? 'Log in' : 'Create an account'}
            </button>
          </p>
        </div>

        <div className="feature-grid">
          {FEATURES.map(({ icon, label }) => (
            <div key={label} className="card feature-card">
              <span className="feature-icon" aria-hidden="true">{icon}</span>
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <main className="app-root">
        <header className="card topbar">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">🩺</span>
            <div className="brand-text">
              <h1>MediQ AI</h1>
              <p>Educational guidance, symptom triage and report summaries</p>
            </div>
          </div>
          <span className="topbar-user">Signed in as <strong>{user}</strong></span>
          <ThemeToggle theme={theme} setTheme={setTheme} />
          <button type="button" className="btn btn-ghost btn-sm" onClick={handleLogout}>Log out</button>
        </header>

        <div className="app-grid">
          <div className="main-panel">
            <SafetyDisclaimer />
            <Chat messages={messages} setMessages={setMessages} draft={draft} setDraft={setDraft} />
          </div>

          <aside className="side-panel">
            <History messages={messages} onReplay={(userMessage) => setDraft(userMessage)} />
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;
