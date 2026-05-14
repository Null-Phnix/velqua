/**
 * File import — drag & drop, file selection, upload with progress stages.
 */

import { API_BASE } from '../api.js';
import { loadFacts } from './facts.js';
import { loadStatus } from './status.js';
import type { SseImportEvent } from '../types.js';

const dropZone = document.getElementById('dropZone') as HTMLElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const status = document.getElementById('status') as HTMLElement;
const statusText = document.getElementById('statusText') as HTMLElement;
const progressFill = document.getElementById('progressFill') as HTMLElement;
const importStats = document.getElementById('importStats') as HTMLElement;

export async function handleFile(file: File): Promise<void> {
    if (!file || !file.name.endsWith('.json')) {
        showError('Please select a JSON file');
        return;
    }

    status.style.display = 'block';
    status.classList.remove('error');
    importStats.style.display = 'none';

    const oldRetry = status.querySelector('.retry-button');
    if (oldRetry) oldRetry.remove();

    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    statusText.textContent = 'Uploading ' + file.name + ' (' + sizeMB + 'MB)...';
    statusText.className = 'status-text status-uploading';
    progressFill.style.width = '5%';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_BASE + '/import/smart/stream', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let detail = response.statusText;
            try { const err = await response.json() as { detail?: string }; detail = err.detail ?? detail; } catch (_) {}
            throw new Error(detail);
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                let data: SseImportEvent;
                try { data = JSON.parse(line.slice(6)) as SseImportEvent; } catch (_) { continue; }

                progressFill.style.width = data.pct + '%';
                statusText.textContent = data.msg ?? data.stage;

                if (data.stage === 'error') {
                    showError(data.msg ?? 'Unknown error');
                    return;
                }

                if (data.stage === 'complete') {
                    progressFill.style.width = '100%';
                    statusText.className = 'status-text status-success';
                    let msg = 'Done: ' + data.msg;
                    if ((data.fiction ?? 0) > 0) msg += ' (' + data.fiction + ' fiction filtered)';
                    if ((data.duplicates ?? 0) > 0) msg += ' (' + data.duplicates + ' duplicates skipped)';
                    statusText.textContent = msg;

                    importStats.style.display = 'grid';
                    (document.getElementById('factsExtracted') as HTMLElement).textContent = String(data.extracted ?? 0);
                    (document.getElementById('factsStored') as HTMLElement).textContent = String(data.stored ?? 0);
                    (document.getElementById('fictionFiltered') as HTMLElement).textContent = String(data.fiction ?? 0);
                    (document.getElementById('duplicatesSkipped') as HTMLElement).textContent = String(data.duplicates ?? 0);

                    void loadFacts();
                    void loadStatus();
                }
            }
        }

    } catch (error) {
        status.classList.add('error');
        let errorMessage: string;
        const msg = (error as Error).message;
        if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
            errorMessage = 'Server not running. Start the backend with: python backend/server.py';
        } else if (msg.includes('413')) {
            errorMessage = 'File too large (max 100MB)';
        } else {
            errorMessage = msg;
        }
        statusText.textContent = 'Error: ' + errorMessage;
        statusText.className = 'status-text status-error';
        progressFill.style.width = '0%';

        if (!status.querySelector('.retry-button')) {
            const retryBtn = document.createElement('button');
            retryBtn.className = 'retry-button';
            retryBtn.textContent = 'Retry Upload';
            retryBtn.onclick = () => {
                status.querySelector('.retry-button')?.remove();
                void handleFile(file);
            };
            status.appendChild(retryBtn);
        }
    }
}

function showError(message: string): void {
    status.style.display = 'block';
    status.classList.add('error');
    statusText.textContent = 'Error: ' + message;
    statusText.className = 'status-text status-error';
    statusText.style.whiteSpace = 'pre-line';
    progressFill.style.width = '0%';
}

/** Set up drag-and-drop and file input listeners. */
export function init(): void {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e: DragEvent) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));

    dropZone.addEventListener('drop', async (e: DragEvent) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        if (e.dataTransfer?.files[0]) await handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', async (e: Event) => {
        const target = e.target as HTMLInputElement;
        if (target.files?.[0]) await handleFile(target.files[0]);
    });
}
