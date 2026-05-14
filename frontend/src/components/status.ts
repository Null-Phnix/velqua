/**
 * Status tab — system health, proxy status, contradictions, backups, import history.
 */

import { API_BASE, apiFetch } from '../api.js';
import { showConfirm, showToast, createSpinner } from './modal.js';
import { loadFacts } from './facts.js';
import type {
    HealthData, FactStats, ProxyStatus, ContradictionsResponse,
    BackupListResponse, BackupCreateResponse, ImportHistoryResponse,
    ExportResponse, CompactResponse, PreviewData
} from '../types.js';

export async function loadStatus(): Promise<void> {
    try {
        const response = await apiFetch(API_BASE + '/health');
        const data = await response.json() as HealthData;

        (document.getElementById('totalFacts') as HTMLElement).textContent = String(data.facts_count);
        (document.getElementById('dbSize') as HTMLElement).textContent = data.database_size_mb + ' MB';

        _updateFirstRunBanner(data.facts_count === 0);

        try {
            const statsR = await apiFetch(API_BASE + '/facts/stats');
            const stats = await statsR.json() as FactStats;
            const learnedEl = document.getElementById('totalLearned');
            if (learnedEl) learnedEl.textContent = String(stats.total ?? 0);
        } catch (_) {}

    } catch (_) {
        (document.getElementById('totalFacts') as HTMLElement).textContent = '--';
        (document.getElementById('dbSize') as HTMLElement).textContent = 'offline';
    }
}

let _bannerDismissed = false;

function _updateFirstRunBanner(show: boolean): void {
    const banner = document.getElementById('firstRunBanner');
    if (!banner) return;
    banner.style.display = (show && !_bannerDismissed) ? 'flex' : 'none';
}

(function _initBanner() {
    const importBtn = document.getElementById('firstRunImportBtn');
    const dismissBtn = document.getElementById('firstRunDismissBtn');
    if (importBtn) {
        importBtn.addEventListener('click', () => {
            (document.querySelector('[data-tab="import"]') as HTMLElement | null)?.click();
        });
    }
    if (dismissBtn) {
        dismissBtn.addEventListener('click', () => {
            _bannerDismissed = true;
            const banner = document.getElementById('firstRunBanner');
            if (banner) banner.style.display = 'none';
        });
    }
})();

export async function loadProxyHealth(): Promise<void> {
    const container = document.getElementById('proxyHealth') as HTMLElement;
    container.textContent = '';

    try {
        const proxyResponse = await apiFetch(API_BASE + '/proxy-status');
        const proxyData = await proxyResponse.json() as ProxyStatus;

        if (proxyData.status === 'offline') {
            throw new Error(proxyData.error ?? 'Proxy not running');
        }

        interface HealthRow { label: string; value: string; status: string; }
        const rows: HealthRow[] = [
            { label: 'Proxy', value: 'Running', status: 'ok' },
            { label: 'Ollama Backend', value: proxyData.backends?.['ollama'] ?? 'localhost:11434', status: 'ok' },
            { label: 'Memory Budget', value: (proxyData.memory_config?.budget ?? 'minimal') + ' (' + (proxyData.memory_config?.max_tokens ?? 200) + ' tokens)', status: 'ok' },
            { label: 'Vector Retrieval', value: proxyData.vector_retrieval ? 'Enabled' : 'Disabled', status: proxyData.vector_retrieval ? 'ok' : 'warn' },
            { label: 'Embedding Model', value: proxyData.model_cached == null ? 'N/A' : proxyData.model_cached ? 'Ready' : 'Will download ~90MB on first chat', status: proxyData.model_cached ? 'ok' : 'warn' },
            { label: 'Auto-Learning', value: proxyData.auto_learning?.enabled ? 'On (' + proxyData.auto_learning.facts_learned + ' learned)' : 'Disabled', status: proxyData.auto_learning?.enabled ? 'ok' : 'warn' },
        ];

        if ((proxyData.auto_learning?.facts_pending ?? 0) > 0) {
            rows.push({ label: 'Pending Review', value: proxyData.auto_learning!.facts_pending + ' facts', status: 'warn' });
        }

        rows.forEach(row => {
            const div = document.createElement('div');
            div.className = 'health-row';
            const labelDiv = document.createElement('span');
            labelDiv.className = 'health-label';
            const indicator = document.createElement('span');
            indicator.className = 'health-indicator health-' + row.status;
            labelDiv.appendChild(indicator);
            labelDiv.appendChild(document.createTextNode(row.label));
            const valueDiv = document.createElement('span');
            valueDiv.className = 'health-value';
            valueDiv.textContent = row.value;
            div.appendChild(labelDiv);
            div.appendChild(valueDiv);
            container.appendChild(div);
        });

    } catch (_) {
        const row = document.createElement('div');
        row.className = 'health-row';
        const label = document.createElement('span');
        label.className = 'health-label';
        const indicator = document.createElement('span');
        indicator.className = 'health-indicator health-error';
        label.appendChild(indicator);
        label.appendChild(document.createTextNode('Proxy'));
        const value = document.createElement('span');
        value.className = 'health-value';
        value.style.color = '#f87171';
        value.textContent = 'Not running';
        row.appendChild(label);
        row.appendChild(value);
        container.appendChild(row);

        const hint = document.createElement('p');
        hint.style.color = '#888';
        hint.style.fontSize = '0.85em';
        hint.style.marginTop = '10px';
        hint.textContent = 'Start proxy: python backend/proxy.py';
        container.appendChild(hint);
    }
}

export async function loadContradictions(): Promise<void> {
    const container = document.getElementById('contradictionsContainer') as HTMLElement;
    container.textContent = '';
    container.appendChild(createSpinner('Scanning for contradictions...'));

    try {
        const r = await apiFetch(API_BASE + '/facts/contradictions');
        const data = await r.json() as ContradictionsResponse;
        container.textContent = '';

        if (data.error) {
            const errP = document.createElement('p');
            errP.style.color = '#888';
            errP.textContent = data.error;
            container.appendChild(errP);
            return;
        }

        if (data.count === 0) {
            const okP = document.createElement('p');
            okP.style.color = '#4ade80';
            okP.textContent = 'No contradictions found. Your knowledge base is consistent.';
            container.appendChild(okP);
            return;
        }

        const summary = document.createElement('p');
        summary.style.color = '#fbbf24';
        summary.style.marginBottom = '10px';
        summary.textContent = data.count + ' contradiction' + (data.count !== 1 ? 's' : '') + ' found:';
        container.appendChild(summary);

        data.contradictions.forEach(c => {
            const pair = document.createElement('div');
            pair.className = 'contradiction-pair';
            const factsRow = document.createElement('div');
            factsRow.className = 'contradiction-facts';
            const factA = document.createElement('div');
            factA.className = 'contradiction-fact';
            factA.textContent = c.fact_a.content;
            const factB = document.createElement('div');
            factB.className = 'contradiction-fact';
            factB.textContent = c.fact_b.content;
            factsRow.appendChild(factA);
            factsRow.appendChild(factB);
            pair.appendChild(factsRow);

            const meta = document.createElement('div');
            meta.className = 'contradiction-meta';
            const info = document.createElement('span');
            info.textContent = c.type + ' (' + (c.confidence * 100).toFixed(0) + '% confidence)';
            meta.appendChild(info);

            const btns = document.createElement('div');
            btns.style.display = 'flex';
            btns.style.gap = '6px';
            const supersedeA = document.createElement('button');
            supersedeA.className = 'supersede-btn';
            supersedeA.textContent = 'Supersede A';
            supersedeA.addEventListener('click', () => supersedeFact(c.fact_a.id));
            const supersedeB = document.createElement('button');
            supersedeB.className = 'supersede-btn';
            supersedeB.textContent = 'Supersede B';
            supersedeB.addEventListener('click', () => supersedeFact(c.fact_b.id));
            btns.appendChild(supersedeA);
            btns.appendChild(supersedeB);
            meta.appendChild(btns);
            pair.appendChild(meta);
            container.appendChild(pair);
        });
    } catch (error) {
        container.textContent = '';
        const errP = document.createElement('p');
        errP.style.color = '#f87171';
        errP.textContent = 'Scan failed: ' + (error as Error).message;
        container.appendChild(errP);
    }
}

async function supersedeFact(factId: string): Promise<void> {
    showConfirm('Mark this fact as superseded (outdated)?', async () => {
        try {
            await apiFetch(API_BASE + '/facts/' + factId + '/supersede', { method: 'POST' });
            showToast('Fact marked as superseded', 'success');
            void loadContradictions();
        } catch (error) {
            showToast('Failed: ' + (error as Error).message, 'error');
        }
    }, { title: 'Supersede Fact', confirmText: 'Supersede', danger: true });
}

export async function previewMemory(): Promise<void> {
    const query = (document.getElementById('previewQuery') as HTMLInputElement).value.trim();
    if (!query) return;

    const btn = document.getElementById('previewBtn') as HTMLButtonElement;
    const result = document.getElementById('previewResult') as HTMLElement;
    const meta = document.getElementById('previewMeta') as HTMLElement;
    const factsDiv = document.getElementById('previewFacts') as HTMLElement;
    const contextDiv = document.getElementById('previewContext') as HTMLElement;
    const toggleBtn = document.getElementById('previewContextToggle') as HTMLButtonElement;

    btn.disabled = true;
    btn.textContent = '...';
    result.style.display = 'none';

    const PROXY_BASE = 'http://localhost:11435';
    try {
        const r = await fetch(PROXY_BASE + '/proxy/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });
        if (!r.ok) throw new Error('Proxy not running (start backend/proxy.py)');
        const data = await r.json() as PreviewData;

        result.style.display = 'block';
        meta.textContent = `${data.facts_injected} of ${data.facts_available} facts injected · ${data.tokens_used}/${data.token_budget} tokens · ${data.search_mode}${data.query_category ? ' · topic: ' + data.query_category : ''}`;

        factsDiv.textContent = '';
        if (data.injected.length === 0) {
            const empty = document.createElement('p');
            empty.style.color = '#888';
            empty.style.fontSize = '0.85em';
            empty.textContent = 'No relevant facts found for this query.';
            factsDiv.appendChild(empty);
        } else {
            data.injected.forEach((f, i) => {
                const row = document.createElement('div');
                row.style.cssText = 'display:flex; justify-content:space-between; align-items:center; padding:5px 8px; background:#0d0d1a; border-radius:4px; border-left:3px solid #667eea;';
                const text = document.createElement('span');
                text.style.cssText = 'font-size:0.85em; color:#ccc; flex:1;';
                text.textContent = (i + 1) + '. ' + f.content;
                const score = document.createElement('span');
                score.style.cssText = 'font-size:0.75em; color:#888; margin-left:10px; white-space:nowrap;';
                score.textContent = (f.score * 100).toFixed(0) + '%' + (f.topic_boost > 1 ? ' ↑' : '');
                row.appendChild(text);
                row.appendChild(score);
                factsDiv.appendChild(row);
            });
        }

        if (data.skipped.length > 0) {
            const skippedLabel = document.createElement('p');
            skippedLabel.style.cssText = 'font-size:0.78em; color:#666; margin-top:6px;';
            skippedLabel.textContent = `+ ${data.skipped.length} more facts over token budget`;
            factsDiv.appendChild(skippedLabel);
        }

        contextDiv.textContent = data.context ?? '(no context would be injected)';
        toggleBtn.onclick = () => {
            const showing = contextDiv.style.display !== 'none';
            contextDiv.style.display = showing ? 'none' : 'block';
            toggleBtn.textContent = showing ? 'Show raw context' : 'Hide raw context';
        };

    } catch (err) {
        result.style.display = 'block';
        meta.textContent = 'Error: ' + (err as Error).message;
        meta.style.color = '#f87171';
        factsDiv.textContent = '';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Preview';
    }
}

export async function compactMemory(): Promise<void> {
    const btn = document.getElementById('compactBtn') as HTMLButtonElement | null;
    const statusDiv = document.getElementById('backupStatus') as HTMLElement | null;
    if (btn) btn.disabled = true;
    if (statusDiv) { statusDiv.textContent = 'Scanning for duplicates...'; statusDiv.style.color = ''; }
    try {
        const r = await apiFetch(API_BASE + '/facts/compact', { method: 'POST' });
        const data = await r.json() as CompactResponse;
        if (statusDiv) {
            statusDiv.textContent = data.message;
            statusDiv.style.color = data.superseded > 0 ? '#4ade80' : '#888';
        }
        showToast(data.message, 'success');
        void loadStatus();
        void loadFacts();
    } catch (error) {
        if (statusDiv) { statusDiv.textContent = 'Compact failed: ' + (error as Error).message; statusDiv.style.color = '#f87171'; }
        showToast('Compact failed: ' + (error as Error).message, 'error');
    } finally {
        if (btn) btn.disabled = false;
    }
}

export async function createBackup(): Promise<void> {
    const statusDiv = document.getElementById('backupStatus') as HTMLElement;
    statusDiv.textContent = 'Creating backup...';
    try {
        const r = await apiFetch(API_BASE + '/backup/create', { method: 'POST' });
        const data = await r.json() as BackupCreateResponse;
        statusDiv.textContent = 'Backup created: ' + data.backup_path + ' (' + data.size_mb + ' MB)';
        statusDiv.style.color = '#4ade80';
        void loadBackupList();
    } catch (error) {
        statusDiv.textContent = 'Backup failed: ' + (error as Error).message;
        statusDiv.style.color = '#f87171';
    }
}

export async function exportFacts(): Promise<void> {
    try {
        const r = await apiFetch(API_BASE + '/export/facts');
        const data = await r.json() as ExportResponse;
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'velqua_facts_export_' + new Date().toISOString().split('T')[0] + '.json';
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        showToast('Exported ' + data.count + ' facts', 'success');
    } catch (error) {
        showToast('Export failed: ' + (error as Error).message, 'error');
    }
}

export async function loadBackupList(): Promise<void> {
    const container = document.getElementById('backupList');
    if (!container) return;
    container.textContent = '';

    try {
        const r = await apiFetch(API_BASE + '/backup/list');
        const data = await r.json() as BackupListResponse;
        const backups = data.backups ?? [];

        if (backups.length === 0) {
            const emptyP = document.createElement('p');
            emptyP.style.color = '#888';
            emptyP.style.fontSize = '0.85em';
            emptyP.textContent = 'No backups yet.';
            container.appendChild(emptyP);
            return;
        }

        backups.forEach(b => {
            const item = document.createElement('div');
            item.className = 'backup-item';
            const info = document.createElement('div');
            const name = document.createElement('span');
            name.textContent = b.filename;
            info.appendChild(name);
            const meta = document.createElement('div');
            meta.className = 'backup-meta';
            meta.textContent = (b.size_mb ?? 0) + ' MB';
            info.appendChild(meta);

            const restoreBtn = document.createElement('button');
            restoreBtn.className = 'restore-btn';
            restoreBtn.textContent = 'Restore';
            restoreBtn.addEventListener('click', () => {
                showConfirm('Restore from backup "' + b.filename + '"? This will replace your current database.', async () => {
                    try {
                        await apiFetch(API_BASE + '/backup/restore/' + encodeURIComponent(b.filename), { method: 'POST' });
                        showToast('Database restored from backup', 'success');
                        void loadStatus();
                        void loadFacts();
                    } catch (err) {
                        showToast('Restore failed: ' + (err as Error).message, 'error');
                    }
                }, { title: 'Restore Backup', confirmText: 'Restore', danger: true });
            });

            item.appendChild(info);
            item.appendChild(restoreBtn);
            container.appendChild(item);
        });
    } catch (_) {}
}

export async function loadImportHistory(): Promise<void> {
    const container = document.getElementById('importHistory') as HTMLElement;
    container.textContent = '';

    try {
        const r = await apiFetch(API_BASE + '/import/history');
        const data = await r.json() as ImportHistoryResponse;

        if (data.history.length === 0) {
            const emptyP = document.createElement('p');
            emptyP.style.color = '#888';
            emptyP.textContent = 'No imports yet.';
            container.appendChild(emptyP);
            return;
        }

        data.history.forEach(entry => {
            const item = document.createElement('div');
            item.className = 'history-item';

            if (entry.undone) {
                item.style.opacity = '0.5';
                item.style.textDecoration = 'line-through';
            }

            const infoDiv = document.createElement('div');
            const typeSpan = document.createElement('span');
            typeSpan.style.color = '#667eea';
            typeSpan.textContent = entry.file_type.replace('_', ' ');
            infoDiv.appendChild(typeSpan);

            const details = document.createElement('span');
            details.className = 'history-meta';
            let detailText = ' — ' + entry.facts_stored + ' facts stored';
            if (entry.duplicates_skipped > 0)
                detailText += ', ' + entry.duplicates_skipped + ' dupes';
            details.textContent = detailText;
            infoDiv.appendChild(details);

            const timeSpan = document.createElement('div');
            timeSpan.className = 'history-meta';
            timeSpan.textContent = new Date(entry.timestamp * 1000).toLocaleString();
            infoDiv.appendChild(timeSpan);

            item.appendChild(infoDiv);

            if (!entry.undone && entry.fact_ids && entry.fact_ids.length > 0) {
                const undoBtn = document.createElement('button');
                undoBtn.className = 'undo-btn';
                undoBtn.textContent = 'Undo';
                undoBtn.addEventListener('click', () => {
                    showConfirm('Undo this import? ' + entry.facts_stored + ' facts will be deleted.', async () => {
                        try {
                            await apiFetch(API_BASE + '/import/undo/' + entry.batch_id, { method: 'POST' });
                            void loadImportHistory();
                            void loadStatus();
                            showToast('Import undone — ' + entry.facts_stored + ' facts removed', 'success');
                        } catch (err) {
                            showToast('Undo failed: ' + (err as Error).message, 'error');
                        }
                    }, { title: 'Undo Import', confirmText: 'Undo', danger: true });
                });
                item.appendChild(undoBtn);
            }

            container.appendChild(item);
        });

    } catch (error) {
        container.textContent = '';
        const errP = document.createElement('p');
        errP.style.color = '#f87171';
        errP.textContent = 'Failed to load history: ' + (error as Error).message;
        container.appendChild(errP);
    }
}
