/**
 * Velqua UI Application — Entry Point
 *
 * Imports all component modules, wires event listeners, and starts polling.
 */

import { loadFacts, searchFacts, bulkDeleteSelected, mergeSelected, loadFactTypes } from './components/facts.js';
import { init as initFacts } from './components/facts.js';
import { loadPending, approveAllPending, rejectAllPending } from './components/review.js';
import { loadTimeline } from './components/timeline.js';
import { loadInsights } from './components/insights.js';
import { loadStatus, loadProxyHealth, loadContradictions, createBackup, exportFacts, compactMemory, previewMemory, loadBackupList, loadImportHistory } from './components/status.js';
import { loadSettings, testConnection, saveProvider, removeProvider, updateMemoryBudget, toggleAutoLearning } from './components/settings.js';
import { showLicenseModal, hideLicenseModal, activateLicense, deactivateLicense } from './components/license.js';
import { onboardNext, closeOnboarding, showOnboarding, wizardTestConnection, setWizardProvider } from './components/wizard.js';
import { init as initImport } from './components/import.js';
import { API_BASE, apiFetch } from './api.js';
import { init as initMesh, loadMesh } from './components/mesh.js';

// ============================================================
// Tab Navigation
// ============================================================

function switchTab(tabName: string): void {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
        tab.setAttribute('aria-selected', 'false');
    });

    (document.getElementById(tabName + '-tab') as HTMLElement).classList.add('active');

    document.querySelectorAll('.nav-tab').forEach(btn => {
        if ((btn as HTMLElement).dataset['tab'] === tabName) {
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
        }
    });

    if (tabName === 'facts') { void loadFacts(); void loadFactTypes(); }
    else if (tabName === 'review') void loadPending();
    else if (tabName === 'timeline') void loadTimeline();
    else if (tabName === 'insights') void loadInsights();
    else if (tabName === 'status') { void loadStatus(); void loadProxyHealth(); void loadImportHistory(); void loadBackupList(); }
    else if (tabName === 'settings') void loadSettings();
    else if (tabName === 'mesh') void loadMesh();
}

// ============================================================
// Keyboard Shortcuts
// ============================================================

let helpVisible = false;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;

function showKeyboardHelp(): void {
    if (helpVisible) { hideKeyboardHelp(); return; }

    const overlay = document.createElement('div');
    overlay.id = 'keyboardHelp';
    overlay.className = 'help-overlay';
    overlay.onclick = hideKeyboardHelp;

    const content = document.createElement('div');
    content.className = 'help-content';
    content.onclick = (e: MouseEvent) => e.stopPropagation();

    const title = document.createElement('h3');
    title.textContent = 'Keyboard Shortcuts';
    content.appendChild(title);

    const shortcuts: [string, string][] = [
        ['Ctrl/Cmd + U', 'Upload file'],
        ['Ctrl/Cmd + Shift + R', 'Refresh facts'],
        ['Escape', 'Close modals'],
        ['?', 'Toggle this help']
    ];

    shortcuts.forEach(([key, desc]) => {
        const row = document.createElement('div');
        row.className = 'help-row';
        row.appendChild(document.createTextNode(desc));
        const kbd = document.createElement('kbd');
        kbd.textContent = key;
        row.appendChild(kbd);
        content.appendChild(row);
    });

    overlay.appendChild(content);
    document.body.appendChild(overlay);
    helpVisible = true;
}

function hideKeyboardHelp(): void {
    const help = document.getElementById('keyboardHelp');
    if (help) help.remove();
    helpVisible = false;
}

document.addEventListener('keydown', (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        fileInput.click();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        void loadFacts();
        void loadStatus();
    }
    if (e.key === 'Escape') {
        const detailOverlay = document.getElementById('factDetailOverlay');
        if (detailOverlay) { detailOverlay.remove(); return; }
        const licenseModal = document.getElementById('licenseModal') as HTMLElement | null;
        if (licenseModal && licenseModal.style.display !== 'none') { licenseModal.style.display = 'none'; return; }
        const modal = document.getElementById('velquaModal') as HTMLElement | null;
        if (modal && modal.style.display !== 'none') { modal.style.display = 'none'; return; }
        closeOnboarding();
        hideKeyboardHelp();
    }
    if (e.key === '?' && !(e.target as HTMLElement).closest('input, textarea')) {
        showKeyboardHelp();
    }
});

// ============================================================
// Pending Badge (polled for live updates)
// ============================================================

let badgePollFailures = 0;
let badgePollInterval: ReturnType<typeof setInterval>;

async function loadPendingBadge(): Promise<void> {
    try {
        const r = await apiFetch(API_BASE + '/review/pending');
        const data = await r.json() as { count: number };
        const badge = document.getElementById('reviewBadge') as HTMLElement;
        if (data.count > 0) {
            badge.textContent = String(data.count);
            badge.style.display = 'inline';
        } else {
            badge.style.display = 'none';
        }

        const pendingEl = document.getElementById('pendingCount');
        if (pendingEl) pendingEl.textContent = String(data.count);
        badgePollFailures = 0;
    } catch (_) {
        badgePollFailures++;
        if (badgePollFailures >= 3) {
            clearInterval(badgePollInterval);
        }
    }
}

// ============================================================
// Initialize Modules
// ============================================================

initImport();
initFacts();
initMesh();

// ============================================================
// Event Listeners
// ============================================================

(document.querySelector('.nav-tabs') as HTMLElement).addEventListener('click', (e: Event) => {
    const tab = (e.target as HTMLElement).closest('.nav-tab') as HTMLElement | null;
    if (tab?.dataset['tab']) switchTab(tab.dataset['tab']);
});

(document.getElementById('onboardBtn1') as HTMLElement).addEventListener('click', () => onboardNext(2));
(document.getElementById('onboardBtn2') as HTMLElement).addEventListener('click', () => onboardNext(3));
(document.getElementById('onboardBtn3') as HTMLElement).addEventListener('click', () => onboardNext(4));
(document.getElementById('onboardBtn4') as HTMLElement).addEventListener('click', () => onboardNext(5));
(document.getElementById('onboardBtn5') as HTMLElement).addEventListener('click', closeOnboarding);
(document.getElementById('wizardTestBtn') as HTMLElement).addEventListener('click', () => void wizardTestConnection());

document.querySelectorAll('.wizard-provider-card').forEach(card => {
    card.addEventListener('click', () => {
        document.querySelectorAll('.wizard-provider-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        setWizardProvider((card as HTMLElement).dataset['provider'] ?? '');
    });
});

(document.getElementById('searchBtn') as HTMLElement).addEventListener('click', () => void searchFacts());
(document.getElementById('bulkDeleteBtn') as HTMLElement).addEventListener('click', () => void bulkDeleteSelected());
(document.getElementById('mergeBtn') as HTMLElement).addEventListener('click', () => void mergeSelected());

(document.getElementById('approveAllBtn') as HTMLElement).addEventListener('click', () => void approveAllPending());
(document.getElementById('rejectAllBtn') as HTMLElement).addEventListener('click', () => void rejectAllPending());

(document.getElementById('meshRefreshAgentsBtn') as HTMLElement).addEventListener('click', () => void loadMesh());

(document.getElementById('previewBtn') as HTMLElement).addEventListener('click', () => void previewMemory());
(document.getElementById('previewQuery') as HTMLElement).addEventListener('keydown', (e: Event) => {
    if ((e as KeyboardEvent).key === 'Enter') void previewMemory();
});
(document.getElementById('compactBtn') as HTMLElement).addEventListener('click', () => void compactMemory());
(document.getElementById('backupBtn') as HTMLElement).addEventListener('click', () => void createBackup());
(document.getElementById('exportBtn') as HTMLElement).addEventListener('click', () => void exportFacts());
(document.getElementById('scanContradictionsBtn') as HTMLElement).addEventListener('click', () => void loadContradictions());

(document.getElementById('testConnectionBtn') as HTMLElement).addEventListener('click', () => void testConnection());
(document.getElementById('saveProviderBtn') as HTMLElement).addEventListener('click', () => void saveProvider(true));
(document.getElementById('removeProviderBtn') as HTMLElement).addEventListener('click', () => void removeProvider());
(document.getElementById('memoryBudget') as HTMLElement).addEventListener('change', () => void updateMemoryBudget());
(document.getElementById('autoLearningToggle') as HTMLElement).addEventListener('change', () => void toggleAutoLearning());
(document.getElementById('toggleKeyVisibility') as HTMLElement).addEventListener('click', () => {
    const input = document.getElementById('providerApiKey') as HTMLInputElement;
    input.type = input.type === 'password' ? 'text' : 'password';
});

(document.getElementById('activateLicenseBtn') as HTMLElement).addEventListener('click', showLicenseModal);
(document.getElementById('deactivateLicenseBtn') as HTMLElement).addEventListener('click', () => void deactivateLicense());
(document.getElementById('licenseModalCancel') as HTMLElement).addEventListener('click', hideLicenseModal);
(document.getElementById('licenseModalActivate') as HTMLElement).addEventListener('click', () => void activateLicense());
(document.getElementById('licenseKeyInput') as HTMLElement).addEventListener('keydown', (e: Event) => {
    if ((e as KeyboardEvent).key === 'Enter') void activateLicense();
});

// ============================================================
// Startup
// ============================================================

if (!localStorage.getItem('velqua_onboarding_done')) {
    showOnboarding();
}

void loadStatus();
void loadPendingBadge();
let statusPollInterval = setInterval(() => void loadStatus(), 10000);
badgePollInterval = setInterval(() => void loadPendingBadge(), 8000);

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        clearInterval(statusPollInterval);
        clearInterval(badgePollInterval);
    } else {
        void loadStatus();
        void loadPendingBadge();
        statusPollInterval = setInterval(() => void loadStatus(), 10000);
        badgePollInterval = setInterval(() => void loadPendingBadge(), 8000);
        badgePollFailures = 0;
    }
});
