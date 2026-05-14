/**
 * Mesh tab — real-time multi-agent coordination dashboard.
 *
 * Connects via WebSocket to /mesh/stream for live updates.
 * Falls back to polling when WebSocket is unavailable.
 */

import { API_BASE, apiFetch } from '../api.js';
import { showToast } from './modal.js';
import type {
    AgentInfo, MeshMemoryEntry, MeshNote,
    MeshAgentsResponse, MeshMemoryResponse, MeshNotesResponse, WsMessage, WsSnapshotData
} from '../types.js';

const MESH_BASE = API_BASE;
let _ws: WebSocket | null = null;
let _wsRetries = 0;
const MAX_WS_RETRIES = 3;

// ============================================================
// Init
// ============================================================

export function init(): void {
    _connectWebSocket();
    _wireNoteForm();
}

export async function loadMesh(): Promise<void> {
    await Promise.all([loadAgents(), loadSharedMemory(), loadNotes()]);
}

// ============================================================
// WebSocket — real-time updates
// ============================================================

function _connectWebSocket(): void {
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

    _ws.onmessage = (evt: MessageEvent) => {
        try {
            const msg = JSON.parse(evt.data as string) as WsMessage;
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

function _extractPort(): string {
    return window.location.port || ((API_BASE.match(/:(\d+)/) || [])[1] || '8765');
}

function _handleWsEvent(msg: WsMessage): void {
    switch (msg.type) {
        case 'snapshot': {
            const data = msg.data as WsSnapshotData;
            _renderAgents(data.agents ?? []);
            _renderMemoryFeed(data.memory ?? []);
            _renderNotes(data.notes ?? []);
            break;
        }
        case 'memory_written':
            _prependMemoryEntry(msg.data as MeshMemoryEntry);
            break;
        case 'note_posted':
            _prependNote(msg.data as MeshNote);
            break;
        case 'ping':
            _updateAgentCount((msg.data as { active_agents: number }).active_agents);
            break;
        default:
            break;
    }
}

function _fallbackToPolling(): void {
    void loadMesh();
    setInterval(() => void loadMesh(), 10000);
}

// ============================================================
// Agent Cards
// ============================================================

export async function loadAgents(): Promise<void> {
    try {
        const r = await apiFetch(MESH_BASE + '/mesh/agents?active_only=true');
        const data = await r.json() as MeshAgentsResponse;
        _renderAgents(data.agents ?? []);
    } catch (_) {
        const el = document.getElementById('meshAgentCards');
        if (el) el.textContent = 'Could not load agents.';
    }
}

function _renderAgents(agents: AgentInfo[]): void {
    const container = document.getElementById('meshAgentCards');
    if (!container) return;
    container.textContent = '';

    const count = document.getElementById('meshAgentCount');
    if (count) count.textContent = String(agents.length);

    if (agents.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'mesh-empty';
        empty.textContent = 'No active agents. Connect an app through localhost:11435 to see it appear here.';
        container.appendChild(empty);
        return;
    }

    agents.forEach(agent => container.appendChild(_makeAgentCard(agent)));
}

function _makeAgentCard(agent: AgentInfo): HTMLElement {
    const card = document.createElement('div');
    card.className = 'mesh-agent-card' + (agent.is_active ? ' mesh-agent-active' : '');
    card.dataset['agentId'] = agent.id;

    const header = document.createElement('div');
    header.className = 'mesh-agent-header';

    const dot = document.createElement('span');
    dot.className = 'mesh-dot' + (agent.is_active ? ' mesh-dot-active' : '');
    header.appendChild(dot);

    const name = document.createElement('strong');
    name.className = 'mesh-agent-name';
    name.textContent = agent.name;
    header.appendChild(name);

    const ago = document.createElement('span');
    ago.className = 'text-muted text-xs';
    ago.textContent = _formatAgo(agent.last_seen_ago);
    header.appendChild(ago);

    card.appendChild(header);

    if (agent.current_task) {
        const task = document.createElement('div');
        task.className = 'mesh-agent-task text-sm';
        task.textContent = agent.current_task.slice(0, 120) + (agent.current_task.length > 120 ? '…' : '');
        card.appendChild(task);
    }

    return card;
}

function _updateAgentCount(n: number): void {
    const el = document.getElementById('meshAgentCount');
    if (el) el.textContent = String(n);
}

// ============================================================
// Shared Memory Feed
// ============================================================

export async function loadSharedMemory(): Promise<void> {
    try {
        const r = await apiFetch(MESH_BASE + '/mesh/memory?limit=30');
        const data = await r.json() as MeshMemoryResponse;
        _renderMemoryFeed(data.entries ?? []);
    } catch (_) {}
}

function _renderMemoryFeed(entries: MeshMemoryEntry[]): void {
    const container = document.getElementById('meshMemoryFeed');
    if (!container) return;
    container.textContent = '';

    if (entries.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'text-muted text-sm';
        empty.textContent = 'No shared memory entries yet. Agents write here via POST /mesh/memory.';
        container.appendChild(empty);
        return;
    }

    entries.forEach(entry => container.appendChild(_makeMemoryEntry(entry)));
}

function _prependMemoryEntry(entry: MeshMemoryEntry): void {
    const container = document.getElementById('meshMemoryFeed');
    if (!container) return;
    const el = _makeMemoryEntry(entry);
    el.style.animation = 'fadeIn 0.3s ease';
    container.insertBefore(el, container.firstChild);
}

function _makeMemoryEntry(entry: MeshMemoryEntry): HTMLElement {
    const div = document.createElement('div');
    div.className = 'mesh-memory-entry';

    const meta = document.createElement('div');
    meta.className = 'mesh-entry-meta text-xs text-muted';
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
    content.className = 'mesh-entry-content text-sm';
    content.textContent = entry.content;

    div.appendChild(meta);
    div.appendChild(content);
    return div;
}

// ============================================================
// Noteboard
// ============================================================

export async function loadNotes(): Promise<void> {
    try {
        const r = await apiFetch(MESH_BASE + '/mesh/notes?limit=30');
        const data = await r.json() as MeshNotesResponse;
        _renderNotes(data.notes ?? []);
    } catch (_) {}
}

function _renderNotes(notes: MeshNote[]): void {
    const container = document.getElementById('meshNoteboard');
    if (!container) return;
    container.textContent = '';

    if (notes.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'text-muted text-sm';
        empty.textContent = 'No notes yet. Agents can leave notes for each other via POST /mesh/notes.';
        container.appendChild(empty);
        return;
    }

    notes.forEach(note => container.appendChild(_makeNoteEl(note)));
}

function _prependNote(note: MeshNote): void {
    const container = document.getElementById('meshNoteboard');
    if (!container) return;
    const el = _makeNoteEl(note);
    el.style.animation = 'fadeIn 0.3s ease';
    container.insertBefore(el, container.firstChild);
}

function _makeNoteEl(note: MeshNote): HTMLElement {
    const div = document.createElement('div');
    div.className = 'mesh-note' + (note.read ? '' : ' mesh-note-unread');

    const header = document.createElement('div');
    header.className = 'mesh-note-header text-xs text-muted';
    header.textContent = `${note.from_agent} → ${note.to_agent}`;

    const ts = document.createElement('span');
    ts.className = 'ml-8';
    ts.textContent = new Date(note.timestamp * 1000).toLocaleTimeString();
    header.appendChild(ts);

    if (!note.read) {
        const badge = document.createElement('span');
        badge.className = 'mesh-unread-badge';
        badge.textContent = 'NEW';
        header.appendChild(badge);

        const markRead = document.createElement('button');
        markRead.className = 'btn-xs ml-8';
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
    content.className = 'mesh-note-content text-sm';
    content.textContent = note.content;

    div.appendChild(header);
    div.appendChild(content);
    return div;
}

function _wireNoteForm(): void {
    const postBtn = document.getElementById('meshPostNoteBtn');
    if (!postBtn) return;

    postBtn.addEventListener('click', async () => {
        const from = (document.getElementById('meshNoteFrom') as HTMLInputElement | null)?.value.trim();
        const to = (document.getElementById('meshNoteTo') as HTMLInputElement | null)?.value.trim();
        const content = (document.getElementById('meshNoteContent') as HTMLTextAreaElement | null)?.value.trim();

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
            const contentEl = document.getElementById('meshNoteContent') as HTMLTextAreaElement | null;
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

function _setConnectionStatus(status: 'live' | 'reconnecting' | 'polling'): void {
    const el = document.getElementById('meshConnectionStatus');
    if (!el) return;
    const labels: Record<string, string> = { live: 'Live', reconnecting: 'Reconnecting…', polling: 'Polling' };
    const colors: Record<string, string> = { live: '#4ade80', reconnecting: '#fb923c', polling: '#94a3b8' };
    el.textContent = labels[status] ?? status;
    el.style.color = colors[status] ?? '#aaa';
}

function _formatAgo(seconds: number): string {
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
}
