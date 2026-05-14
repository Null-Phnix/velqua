/**
 * Activity log — chronological feed of system events.
 * Uses createElement throughout to avoid innerHTML XSS.
 */
import { API_BASE, apiFetch } from '../api.js';
import { createEmptyState } from './modal.js';

const ICONS = {
  fact_learned:   '\u{1F9E0}',
  fact_approved:  '\u2705',
  fact_rejected:  '\u274C',
  fact_deleted:   '\u{1F5D1}',
  fact_merged:    '\u{1F500}',
  fact_edited:    '\u270F\uFE0F',
  import_completed: '\u{1F4E5}',
  import_failed:  '\u26A0\uFE0F',
  backup_created: '\u{1F4BE}',
  backup_restored: '\u{1F504}',
  provider_changed: '\u{1F517}',
  agent_connected: '\u{1F916}',
  agent_disconnected: '\u{1F6AB}',
  system_started: '\u26A1',
};

const TYPE_LABELS = {
  fact_learned:   'Fact Learned',
  fact_approved:  'Approved',
  fact_rejected:  'Rejected',
  fact_deleted:   'Deleted',
  fact_merged:    'Merged',
  fact_edited:    'Edited',
  import_completed: 'Import',
  import_failed:  'Import Failed',
  backup_created: 'Backup',
  backup_restored: 'Restore',
  provider_changed: 'Provider',
  agent_connected: 'Agent',
  agent_disconnected: 'Agent Left',
  system_started: 'System',
};

let currentPage = 0;
const PAGE_SIZE = 50;
let currentFilter = '';

export function init() {
  const filterSelect = document.getElementById('activityFilter');
  const refreshBtn = document.getElementById('activityRefreshBtn');
  const prevBtn = document.getElementById('activityPrevBtn');
  const nextBtn = document.getElementById('activityNextBtn');

  if (filterSelect) {
    filterSelect.addEventListener('change', () => {
      currentFilter = filterSelect.value;
      currentPage = 0;
      loadActivity();
    });
  }
  if (refreshBtn) refreshBtn.addEventListener('click', () => loadActivity());
  if (prevBtn) prevBtn.addEventListener('click', () => { if (currentPage > 0) { currentPage--; loadActivity(); } });
  if (nextBtn) nextBtn.addEventListener('click', () => { currentPage++; loadActivity(); });
}

export async function loadActivity() {
  const feed = document.getElementById('activityFeed');
  const pageInfo = document.getElementById('activityPageInfo');
  const prevBtn = document.getElementById('activityPrevBtn');
  const nextBtn = document.getElementById('activityNextBtn');
  if (!feed) return;

  const offset = currentPage * PAGE_SIZE;
  let url = `${API_BASE}/activity?limit=${PAGE_SIZE}&offset=${offset}`;
  if (currentFilter) url += `&event_type=${encodeURIComponent(currentFilter)}`;

  try {
    const resp = await apiFetch(url);
    const data = await resp.json();
    renderTimeline(data.events, feed);

    const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));
    if (pageInfo) pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
    if (prevBtn) prevBtn.disabled = currentPage === 0;
    if (nextBtn) nextBtn.disabled = offset + PAGE_SIZE >= data.total;
  } catch (err) {
    feed.textContent = '';
    const errDiv = document.createElement('div');
    errDiv.className = 'activity-empty';
    errDiv.textContent = 'Could not load activity log.';
    feed.appendChild(errDiv);
  }
}

function renderTimeline(events, container) {
  container.textContent = '';

  if (!events || events.length === 0) {
    container.appendChild(createEmptyState({
      icon: '<svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
      heading: 'No activity yet',
      text: 'Events will appear here as you use Velqua \u2014 imports, learned facts, provider changes, and more.',
      ctaLabel: 'Import your first memory',
      ctaAction: () => document.querySelector('.nav-pill[data-page="home"]')?.click(),
    }));
    return;
  }

  let lastDateStr = '';

  events.forEach(event => {
    const dateStr = formatDate(event.timestamp);

    if (dateStr !== lastDateStr) {
      lastDateStr = dateStr;
      const dateSep = document.createElement('div');
      dateSep.className = 'activity-date-separator';
      const dateLabel = document.createElement('span');
      dateLabel.textContent = dateStr;
      dateSep.appendChild(dateLabel);
      container.appendChild(dateSep);
    }

    const item = document.createElement('div');
    item.className = 'activity-item';

    const iconEl = document.createElement('div');
    iconEl.className = 'activity-icon';
    iconEl.textContent = ICONS[event.event_type] || '\u2022';

    const body = document.createElement('div');
    body.className = 'activity-body';

    const header = document.createElement('div');
    header.className = 'activity-header';

    const badge = document.createElement('span');
    badge.className = `activity-badge activity-badge--${eventCategory(event.event_type)}`;
    badge.textContent = TYPE_LABELS[event.event_type] || event.event_type;

    const time = document.createElement('span');
    time.className = 'activity-time';
    time.textContent = formatTime(event.timestamp);

    header.appendChild(badge);
    header.appendChild(time);

    const title = document.createElement('p');
    title.className = 'activity-title';
    title.textContent = event.title;

    body.appendChild(header);
    body.appendChild(title);

    if (event.detail) {
      const detail = document.createElement('p');
      detail.className = 'activity-detail';
      detail.textContent = event.detail;
      body.appendChild(detail);
    }

    item.appendChild(iconEl);
    item.appendChild(body);
    container.appendChild(item);
  });
}

function eventCategory(type) {
  if (type.startsWith('fact_')) return 'fact';
  if (type.startsWith('import')) return 'import';
  if (type.startsWith('backup')) return 'backup';
  if (type.startsWith('provider')) return 'provider';
  if (type.startsWith('agent')) return 'agent';
  return 'system';
}

function formatDate(ts) {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now - d;
  if (diff < 86400000 && d.getDate() === now.getDate()) return 'Today';
  if (diff < 172800000) {
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.getDate() === yesterday.getDate()) return 'Yesterday';
  }
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}
