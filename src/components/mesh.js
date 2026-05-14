/**
 * Mesh tab — real-time multi-agent coordination dashboard.
 *
 * Connects via WebSocket to /mesh/stream for live updates.
 * Falls back to polling when WebSocket is unavailable.
 *
 * Agent cards show: name, colored status dot, last seen,
 * current task, message count. Cards pulse on incoming data.
 */
import { API_BASE, apiFetch } from '../api.js';
import { showToast, createEmptyState } from './modal.js';

const MESH_BASE = API_BASE;
let _ws = null;
let _wsRetries = 0;
const MAX_WS_RETRIES = 3;
const IDLE_THRESHOLD = 60;  // seconds before "idle" (yellow)
const DISCONNECTED_THRESHOLD = 120; // seconds before "disconnected" (red)

// Track agents for targeted card updates
let _agentMap = new Map();

// ============================================================
// Init
// ============================================================
export function init() {
    _connectWebSocket();
    _wireNoteForm();
}

export async function loadMesh() {
    await Promise.all([loadAgents(), loadSharedMemory(), loadNotes()]);
}

// ============================================================
// WebSocket — real-time updates
// ============================================================
function _connectWebSocket() {
    const wsUrl = `ws://127.0.0.1:${_extractPort()}/mesh/stream`;
    try {
        _ws = new WebSocket(wsUrl);
    } catch (_) {
        _fallbackToPolling();
        return;
    }

    _ws.onopen = () => {
        _wsRetries = 0;
        _setConnectionStatus('live');
    };

    _ws.onmessage = (evt) => {
        try {
            const msg = JSON.parse(evt.data);
            _handleWsEvent(msg);
        } catch (_) {}
    };

    _ws.onclose = () => {
        _setConnectionStatus('reconnecting');
        _wsRetries++;
        if (_wsRetries <= MAX_WS_RETRIES) {
            setTimeout(_connectWebSocket, 3000 * _wsRetries);
        } else {
            _setConnectionStatus('polling');
            _fallbackToPolling();
        }
    };

    _ws.onerror = () => {
        _ws = null;
    };
}

function _extractPort() {
    return window.location.port || ((API_BASE.match(/:(\d+)/) || [])[1] || '8765');
}

function _handleWsEvent(msg) {
    switch (msg.type) {
        case 'snapshot': {
            const data = msg.data;
            _renderAgents(data.agents ?? []);
            _renderMemoryFeed(data.memory ?? []);
            _renderNotes(data.notes ?? []);
            break;
        }
        case 'agent_updated':
            _updateSingleAgent(msg.data);
            break;
        case 'memory_written':
            _prependMemoryEntry(msg.data);
            break;
        case 'note_posted':
            _prependNote(msg.data);
            break;
        case 'ping': {
            _updateAgentCount(msg.data.active_agents);
            if (msg.data.agents) {
                _renderAgents(msg.data.agents);
            }
            break;
        }
        default:
            break;
    }
}

function _fallbackToPolling() {
    void loadMesh();
    setInterval(() => void loadMesh(), 10000);
}

// ============================================================
// Agent Cards — glass cards with status, pulse, message count
// ============================================================
export async function loadAgents() {
    try {
        const r = await apiFetch(MESH_BASE + '/mesh/agents?active_only=false');
        const data = await r.json();
        _renderAgents(data.agents ?? []);
    } catch (_) {
        const el = document.getElementById('meshAgentCards');
        if (el) el.textContent = 'Could not load agents.';
    }
}

function _renderAgents(agents) {
    const container = document.getElementById('meshAgentCards');
    if (!container) return;

    // Update count
    const activeCount = agents.filter(a => a.is_active).length;
    const count = document.getElementById('meshAgentCount');
    if (count) count.textContent = String(activeCount);

    if (agents.length === 0) {
        container.textContent = '';
        _agentMap.clear();
        container.appendChild(createEmptyState({
            icon: '<svg viewBox="0 0 24 24"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M5 8v2a4 4 0 0 0 4 4h6a4 4 0 0 0 4-4V8"/><path d="M12 14v2"/></svg>',
            heading: 'No agents connected',
            text: 'Agents appear here as live cards when they connect through the proxy. Point any OpenAI-compatible app at localhost:11435 to get started.',
            ctaLabel: 'Set up your first connection',
            ctaAction: () => document.querySelector('.tab-btn[data-tab="dashboard"]')?.click(),
        }));
        return;
    }

    // Sort: active first, then by last_seen desc
    agents.sort((a, b) => {
        if (a.is_active !== b.is_active) return a.is_active ? -1 : 1;
        return (b.last_seen || 0) - (a.last_seen || 0);
    });

    // Rebuild or update cards
    const existingIds = new Set();
    agents.forEach(agent => {
        existingIds.add(agent.id);
        const existing = _agentMap.get(agent.id);
        if (existing && container.contains(existing.el)) {
            _updateCardContent(existing.el, agent);
            _agentMap.set(agent.id, { el: existing.el, data: agent });
        } else {
            const card = _makeAgentCard(agent);
            _agentMap.set(agent.id, { el: card, data: agent });
        }
    });

    // Remove stale cards
    for (const [id] of _agentMap) {
        if (!existingIds.has(id)) {
            const entry = _agentMap.get(id);
            if (entry?.el?.parentNode) entry.el.remove();
            _agentMap.delete(id);
        }
    }

    // Re-append in sorted order
    container.textContent = '';
    agents.forEach(agent => {
        const entry = _agentMap.get(agent.id);
        if (entry) container.appendChild(entry.el);
    });
}

function _updateSingleAgent(agentData) {
    const entry = _agentMap.get(agentData.id);
    if (entry) {
        _updateCardContent(entry.el, agentData);
        _pulseCard(entry.el);
        _agentMap.set(agentData.id, { el: entry.el, data: agentData });
    } else {
        // New agent — full re-render
        void loadAgents();
    }

    // Update count
    let activeCount = 0;
    for (const [, e] of _agentMap) {
        if (e.data.is_active) activeCount++;
    }
    const count = document.getElementById('meshAgentCount');
    if (count) count.textContent = String(activeCount);
}

function _getStatus(agent) {
    if (!agent.is_active) return 'disconnected';
    if (agent.last_seen_ago > IDLE_THRESHOLD) return 'idle';
    return 'active';
}

function _makeAgentCard(agent) {
    const status = _getStatus(agent);
    const card = document.createElement('div');
    card.className = `mesh-agent-card mesh-status-${status}`;
    card.dataset.agentId = agent.id;

    // Header row: dot + name
    const header = document.createElement('div');
    header.className = 'mesh-card-header';

    const dot = document.createElement('span');
    dot.className = `mesh-status-dot mesh-dot-${status}`;
    dot.setAttribute('aria-label', status);
    header.appendChild(dot);

    const name = document.createElement('span');
    name.className = 'mesh-card-name';
    name.textContent = agent.name;
    header.appendChild(name);

    card.appendChild(header);

    // Task line
    const task = document.createElement('div');
    task.className = 'mesh-card-task';
    if (agent.current_task) {
        task.textContent = agent.current_task.length > 100
            ? agent.current_task.slice(0, 100) + '…'
            : agent.current_task;
    } else {
        task.textContent = 'No active task';
        task.classList.add('mesh-card-task-empty');
    }
    card.appendChild(task);

    // Footer: last seen + message count
    const footer = document.createElement('div');
    footer.className = 'mesh-card-footer';

    const lastSeen = document.createElement('span');
    lastSeen.className = 'mesh-card-lastseen';
    lastSeen.textContent = _formatLastSeen(agent);
    footer.appendChild(lastSeen);

    const msgs = document.createElement('span');
    msgs.className = 'mesh-card-msgs';
    const msgCount = agent.message_count || 0;
    msgs.textContent = `${msgCount} msg${msgCount !== 1 ? 's' : ''}`;
    footer.appendChild(msgs);

    card.appendChild(footer);

    return card;
}

function _updateCardContent(card, agent) {
    const status = _getStatus(agent);

    // Update status class
    card.className = `mesh-agent-card mesh-status-${status}`;

    // Update dot
    const dot = card.querySelector('.mesh-status-dot');
    if (dot) {
        dot.className = `mesh-status-dot mesh-dot-${status}`;
        dot.setAttribute('aria-label', status);
    }

    // Update name
    const name = card.querySelector('.mesh-card-name');
    if (name) name.textContent = agent.name;

    // Update task
    const task = card.querySelector('.mesh-card-task');
    if (task) {
        if (agent.current_task) {
            task.textContent = agent.current_task.length > 100
                ? agent.current_task.slice(0, 100) + '…'
                : agent.current_task;
            task.classList.remove('mesh-card-task-empty');
        } else {
            task.textContent = 'No active task';
            task.classList.add('mesh-card-task-empty');
        }
    }

    // Update last seen
    const lastSeen = card.querySelector('.mesh-card-lastseen');
    if (lastSeen) lastSeen.textContent = _formatLastSeen(agent);

    // Update message count
    const msgs = card.querySelector('.mesh-card-msgs');
    if (msgs) {
        const msgCount = agent.message_count || 0;
        msgs.textContent = `${msgCount} msg${msgCount !== 1 ? 's' : ''}`;
    }
}

function _pulseCard(card) {
    card.classList.remove('mesh-card-pulse');
    // Force reflow to restart animation
    void card.offsetWidth;
    card.classList.add('mesh-card-pulse');
}

function _updateAgentCount(n) {
    const el = document.getElementById('meshAgentCount');
    if (el) el.textContent = String(n);
}

function _formatLastSeen(agent) {
    if (agent.last_seen_ago < 5) return 'Just now';
    if (agent.last_seen_ago < 60) return `${agent.last_seen_ago}s ago`;
    if (agent.last_seen_ago < 3600) return `${Math.floor(agent.last_seen_ago / 60)}m ago`;
    if (agent.last_seen_ago < 86400) return `${Math.floor(agent.last_seen_ago / 3600)}h ago`;
    return `${Math.floor(agent.last_seen_ago / 86400)}d ago`;
}

// ============================================================
// Shared Memory Feed
// ============================================================
export async function loadSharedMemory() {
    try {
        const r = await apiFetch(MESH_BASE + '/mesh/memory?limit=30');
        const data = await r.json();
        _renderMemoryFeed(data.entries ?? []);
    } catch (_) {}
}

function _renderMemoryFeed(entries) {
    const container = document.getElementById('meshMemoryFeed');
    if (!container) return;
    container.textContent = '';
    if (entries.length === 0) {
        container.appendChild(createEmptyState({
            icon: '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
            heading: 'Shared memory is empty',
            text: 'When agents write cross-agent knowledge, it will appear here in real time. Use POST /mesh/memory to publish entries.',
            compact: true,
        }));
        return;
    }
    entries.forEach(entry => container.appendChild(_makeMemoryEntry(entry)));
}

function _prependMemoryEntry(entry) {
    const container = document.getElementById('meshMemoryFeed');
    if (!container) return;
    const el = _makeMemoryEntry(entry);
    el.classList.add('mesh-feed-enter');
    container.insertBefore(el, container.firstChild);
}

function _makeMemoryEntry(entry) {
    const div = document.createElement('div');
    div.className = 'mesh-memory-entry';

    const meta = document.createElement('div');
    meta.className = 'mesh-entry-meta';
    const ts = new Date(entry.timestamp * 1000).toLocaleTimeString();
    meta.textContent = `${entry.agent_id} · ${ts}`;
    if (entry.tags?.length) {
        entry.tags.forEach(tag => {
            const chip = document.createElement('span');
            chip.className = 'mesh-tag';
            chip.textContent = tag;
            meta.appendChild(chip);
        });
    }

    const content = document.createElement('div');
    content.className = 'mesh-entry-content';
    content.textContent = entry.content;

    div.appendChild(meta);
    div.appendChild(content);
    return div;
}

// ============================================================
// Noteboard
// ============================================================
export async function loadNotes() {
    try {
        const r = await apiFetch(MESH_BASE + '/mesh/notes?limit=30');
        const data = await r.json();
        _renderNotes(data.notes ?? []);
    } catch (_) {}
}

function _renderNotes(notes) {
    const container = document.getElementById('meshNoteboard');
    if (!container) return;
    container.textContent = '';
    if (notes.length === 0) {
        container.appendChild(createEmptyState({
            icon: '<svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
            heading: 'No notes yet',
            text: 'Agents can leave messages for each other here. Use the form below to post the first one.',
            compact: true,
        }));
        return;
    }
    notes.forEach(note => container.appendChild(_makeNoteEl(note)));
}

function _prependNote(note) {
    const container = document.getElementById('meshNoteboard');
    if (!container) return;
    const el = _makeNoteEl(note);
    el.classList.add('mesh-feed-enter');
    container.insertBefore(el, container.firstChild);
}

function _makeNoteEl(note) {
    const div = document.createElement('div');
    div.className = 'mesh-note' + (note.read ? '' : ' mesh-note-unread');

    const header = document.createElement('div');
    header.className = 'mesh-note-header';
    header.textContent = `${note.from_agent} → ${note.to_agent}`;

    const ts = document.createElement('span');
    ts.className = 'mesh-note-ts';
    ts.textContent = new Date(note.timestamp * 1000).toLocaleTimeString();
    header.appendChild(ts);

    if (!note.read) {
        const badge = document.createElement('span');
        badge.className = 'mesh-unread-badge';
        badge.textContent = 'NEW';
        header.appendChild(badge);

        const markRead = document.createElement('button');
        markRead.className = 'btn btn-ghost mesh-mark-read';
        markRead.textContent = 'Mark read';
        markRead.addEventListener('click', async () => {
            await apiFetch(MESH_BASE + '/mesh/notes/' + note.id + '/read', { method: 'PUT' });
            div.classList.remove('mesh-note-unread');
            badge.remove();
            markRead.remove();
        });
        header.appendChild(markRead);
    }

    const content = document.createElement('div');
    content.className = 'mesh-note-content';
    content.textContent = note.content;

    div.appendChild(header);
    div.appendChild(content);
    return div;
}

function _wireNoteForm() {
    const postBtn = document.getElementById('meshPostNoteBtn');
    if (!postBtn) return;

    postBtn.addEventListener('click', async () => {
        const from = document.getElementById('meshNoteFrom')?.value.trim();
        const to = document.getElementById('meshNoteTo')?.value.trim();
        const content = document.getElementById('meshNoteContent')?.value.trim();
        if (!from || !to || !content) {
            showToast('Fill in all note fields.', 'error');
            return;
        }
        try {
            await apiFetch(MESH_BASE + '/mesh/notes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ from_agent: from, to_agent: to, content }),
            });
            const contentEl = document.getElementById('meshNoteContent');
            if (contentEl) contentEl.value = '';
            showToast('Note posted.', 'success');
        } catch (_) {
            showToast('Failed to post note.', 'error');
        }
    });
}

// ============================================================
// Helpers
// ============================================================
function _setConnectionStatus(status) {
    const el = document.getElementById('meshConnectionStatus');
    if (!el) return;
    const labels = { live: 'Live', reconnecting: 'Reconnecting…', polling: 'Polling' };
    const classes = { live: 'mesh-conn-live', reconnecting: 'mesh-conn-reconnecting', polling: 'mesh-conn-polling' };
    el.textContent = labels[status] ?? status;
    el.className = 'mesh-connection-badge ' + (classes[status] ?? '');
}
