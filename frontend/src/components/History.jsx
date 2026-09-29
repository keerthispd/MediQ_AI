import React, { useCallback, useEffect, useState } from 'react';
import { api } from '../api';

// One-line previews show plain text, not Markdown symbols
const plainText = (text) => (text || '').replace(/[#*`_~>]/g, '');

const relativeTime = (iso) => {
  const then = new Date(iso);
  const minutes = Math.round((Date.now() - then.getTime()) / 60000);
  if (Number.isNaN(minutes)) return '';
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 1440) return `${Math.round(minutes / 60)}h ago`;
  if (minutes < 10080) return `${Math.round(minutes / 1440)}d ago`;
  return then.toLocaleDateString();
};

export default function History({ limit = 50, onReplay, messages = [] }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('');

  // Each finished assistant reply means the backend just saved a new exchange, so refresh the list
  const replyCount = messages.filter((m) => m.role === 'assistant' && !m.pending).length;

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/api/history', { params: { limit } });
      setItems(res.data.items || []);
    } catch (e) {
      console.error('Failed to load history', e);
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchHistory();
  }, [replyCount, fetchHistory]);

  const shown = items.filter((it) => {
    if (!filter) return true;
    const f = filter.toLowerCase();
    return (it.user_message || '').toLowerCase().includes(f)
      || (it.assistant_reply || '').toLowerCase().includes(f);
  });

  async function clearHistory() {
    if (!window.confirm('Permanently delete all saved conversations? This cannot be undone.')) return;
    setLoading(true);
    try {
      await api.delete('/api/history');
      setItems([]);
    } catch (e) {
      console.error('Failed to clear history', e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card history-card">
      <div className="history-header">
        <h3>Recent conversations</h3>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={clearHistory}
          disabled={items.length === 0 || loading}
        >
          Clear all
        </button>
      </div>

      {items.length > 0 && (
        <div className="history-search">
          <input
            placeholder="Search conversations…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            aria-label="Search conversation history"
          />
        </div>
      )}

      {/* Only show the loading state on first load, so background refreshes don't make the list jump */}
      {loading && items.length === 0 && <div className="history-empty">Loading…</div>}
      {!loading && items.length === 0 && (
        <div className="history-empty">Your past conversations will appear here.</div>
      )}
      {items.length > 0 && shown.length === 0 && (
        <div className="history-empty">Nothing matches “{filter}”.</div>
      )}

      {shown.length > 0 && (
        <div className="history-list">
          {shown.map((it) => (
            <button
              key={it.id}
              type="button"
              className={`history-item ${it.redflag ? 'flagged' : ''}`}
              onClick={() => onReplay && onReplay(it.user_message, it.assistant_reply)}
              title="Put this question back in the message box"
            >
              <div className="history-time">{relativeTime(it.created_at)}</div>
              <div className="history-message">{it.user_message}</div>
              <div className="history-message assistant-preview">{plainText(it.assistant_reply)}</div>
              {it.redflag && <div className="history-flag">⚠️ {it.redflag_details}</div>}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
