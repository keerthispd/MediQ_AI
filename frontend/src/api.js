import axios from 'axios';

const SESSION_KEY = 'mediq_session';

export function loadSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY)) || {};
  } catch (e) {
    return {};
  }
}

export function saveSession({ token, username }) {
  localStorage.setItem(SESSION_KEY, JSON.stringify({ token, username }));
}

export function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

function authHeaders() {
  const { token } = loadSession();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

let onUnauthorized = () => {};

// Called when the backend rejects the session (e.g. it expired), so the app can return to the login screen
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

function isSessionError(status, url) {
  return status === 401 && !url.includes('/api/auth/');
}

export const api = axios.create();

api.interceptors.request.use((config) => {
  const { Authorization } = authHeaders();
  if (Authorization) config.headers.Authorization = Authorization;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (isSessionError(error.response?.status, error.config?.url || '')) onUnauthorized();
    return Promise.reject(error);
  }
);

// FastAPI's `detail` is a string for our own errors and a list for validation errors
export function errorText(error, fallback) {
  const detail = error?.response?.data?.detail ?? error?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => `${d.loc?.[d.loc.length - 1] ?? 'input'}: ${d.msg}`).join('; ');
  }
  return fallback;
}

// POST a form and call onEvent for each event the backend streams back (one JSON object per line)
export async function streamForm(url, form, onEvent, signal) {
  const response = await fetch(url, { method: 'POST', body: form, headers: authHeaders(), signal });
  if (!response.ok) {
    if (isSessionError(response.status, url)) onUnauthorized();
    const error = new Error(`Request failed with status code ${response.status}`);
    try {
      error.detail = (await response.json()).detail;
    } catch (e) {
      // not a JSON error body
    }
    throw error;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const emitLines = () => {
    const lines = buffer.split('\n');
    buffer = lines.pop();
    lines.filter((line) => line.trim()).forEach((line) => onEvent(JSON.parse(line)));
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    emitLines();
  }
  buffer += decoder.decode() + '\n';
  emitLines();
}
