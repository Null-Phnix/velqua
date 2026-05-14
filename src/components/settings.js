/**
 * Settings tab — provider configuration, memory settings.
 */
import { API_BASE, apiFetch } from '../api.js';
import { showConfirm, showToast } from './modal.js';
import { loadLicenseStatus } from './license.js';
const PROVIDER_DEFS = [
    { name: 'ollama', label: 'Ollama', desc: 'Local inference', color: '#4a9eff', needsKey: false },
    { name: 'openai', label: 'OpenAI', desc: 'GPT-4o, GPT-4', color: '#10a37f', needsKey: true },
    { name: 'anthropic', label: 'Anthropic', desc: 'Claude models', color: '#d97706', needsKey: true },
    { name: 'groq', label: 'Groq', desc: 'Fast inference', color: '#8b5cf6', needsKey: true },
    { name: 'local_openai', label: 'Local (OpenAI)', desc: 'llama.cpp, vLLM', color: '#6b7280', needsKey: false },
];
let selectedProvider = null;
export async function loadSettings() {
    try {
        const [settingsRes, providersRes] = await Promise.all([
            apiFetch(API_BASE + '/settings'),
            apiFetch(API_BASE + '/settings/providers'),
        ]);
        const settings = await settingsRes.json();
        const provData = await providersRes.json();
        renderProviderCards(provData.providers, settings.active_provider);
        renderMemorySettings(settings);
        void loadLicenseStatus();
    }
    catch (err) {
        showToast('Failed to load settings: ' + err.message, 'error');
    }
}
function renderProviderCards(providers, activeProvider) {
    const container = document.getElementById('providerCards');
    container.innerHTML = '';
    PROVIDER_DEFS.forEach(def => {
        const configured = providers.find(p => p.name === def.name);
        const isActive = activeProvider === def.name;
        const card = document.createElement('div');
        card.className = 'provider-card' + (isActive ? ' active' : '') + (configured ? ' configured' : '');
        card.style.borderColor = isActive ? def.color : '';
        const header = document.createElement('div');
        header.className = 'provider-card-header';
        const dot = document.createElement('span');
        dot.className = 'provider-dot';
        dot.style.backgroundColor = def.color;
        header.appendChild(dot);
        const nameEl = document.createElement('strong');
        nameEl.textContent = def.label;
        header.appendChild(nameEl);
        if (isActive) {
            const badge = document.createElement('span');
            badge.className = 'provider-active-badge';
            badge.textContent = 'Active';
            header.appendChild(badge);
        }
        card.appendChild(header);
        const desc = document.createElement('div');
        desc.className = 'provider-card-desc';
        desc.textContent = def.desc;
        card.appendChild(desc);
        if (configured?.has_api_key) {
            const keyStatus = document.createElement('div');
            keyStatus.className = 'provider-key-status';
            keyStatus.textContent = 'API key configured';
            card.appendChild(keyStatus);
        }
        card.addEventListener('click', () => openProviderConfig(def, configured ?? null));
        container.appendChild(card);
    });
}
function openProviderConfig(def, existing) {
    selectedProvider = def.name;
    const configCard = document.getElementById('providerConfigCard');
    configCard.style.display = 'block';
    document.getElementById('providerConfigTitle').textContent = 'Configure ' + def.label;
    const badge = document.getElementById('providerBadge');
    badge.textContent = def.label;
    badge.style.backgroundColor = def.color;
    const keyInput = document.getElementById('providerApiKey');
    keyInput.value = '';
    keyInput.type = 'password';
    keyInput.placeholder = def.needsKey ? 'Enter API key...' : 'Not required';
    keyInput.disabled = !def.needsKey;
    const urlInput = document.getElementById('providerBaseUrl');
    urlInput.value = existing?.base_url ?? '';
    urlInput.placeholder =
        def.name === 'ollama' ? 'http://localhost:11434' :
            def.name === 'openai' ? 'https://api.openai.com' :
                def.name === 'anthropic' ? 'https://api.anthropic.com' :
                    def.name === 'groq' ? 'https://api.groq.com/openai' :
                        'http://localhost:8080';
    const modelSelect = document.getElementById('providerModel');
    modelSelect.innerHTML = '<option value="">Select model...</option>';
    if (existing?.models) {
        existing.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === (existing.default_model ?? ''))
                opt.selected = true;
            modelSelect.appendChild(opt);
        });
    }
    const removeBtn = document.getElementById('removeProviderBtn');
    removeBtn.style.display = (def.name === 'ollama') ? 'none' : '';
    document.getElementById('connectionTestResult').innerHTML = '';
    configCard.scrollIntoView({ behavior: 'smooth' });
}
export async function testConnection() {
    if (!selectedProvider)
        return;
    const keyInput = document.getElementById('providerApiKey');
    const urlInput = document.getElementById('providerBaseUrl');
    if (keyInput.value || urlInput.value) {
        await saveProvider(false);
    }
    const resultDiv = document.getElementById('connectionTestResult');
    resultDiv.innerHTML = '<span class="text-muted">Testing connection...</span>';
    try {
        const r = await apiFetch(API_BASE + '/settings/providers/' + selectedProvider + '/test', {
            method: 'POST',
        });
        const data = await r.json();
        if (data.ok) {
            resultDiv.innerHTML = '<span style="color:#4ade80;">Connected! Found ' + data.models.length + ' models.</span>';
            const modelSelect = document.getElementById('providerModel');
            modelSelect.innerHTML = '<option value="">Select model...</option>';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                modelSelect.appendChild(opt);
            });
        }
        else {
            resultDiv.textContent = '';
            const span = document.createElement('span');
            span.style.color = '#f87171';
            span.textContent = 'Connection failed: ' + (data.error ?? 'Unknown error');
            resultDiv.appendChild(span);
        }
    }
    catch (err) {
        resultDiv.textContent = '';
        const span = document.createElement('span');
        span.style.color = '#f87171';
        span.textContent = 'Error: ' + err.message;
        resultDiv.appendChild(span);
    }
}
export async function saveProvider(activate = null) {
    if (!selectedProvider)
        return;
    const body = {
        name: selectedProvider,
        api_key: document.getElementById('providerApiKey').value,
        base_url: document.getElementById('providerBaseUrl').value,
        default_model: document.getElementById('providerModel').value,
        enabled: true,
    };
    try {
        await apiFetch(API_BASE + '/settings/providers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (activate !== false) {
            await apiFetch(API_BASE + '/settings/active-provider', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: selectedProvider }),
            });
            showToast(selectedProvider + ' activated', 'success');
        }
        else {
            showToast(selectedProvider + ' saved', 'success');
        }
        void loadSettings();
    }
    catch (err) {
        showToast('Failed to save provider: ' + err.message, 'error');
    }
}
export async function removeProvider() {
    if (!selectedProvider || selectedProvider === 'ollama')
        return;
    showConfirm('Remove ' + selectedProvider + ' provider?', async () => {
        try {
            await apiFetch(API_BASE + '/settings/providers/' + selectedProvider, {
                method: 'DELETE',
            });
            showToast(selectedProvider + ' removed', 'success');
            document.getElementById('providerConfigCard').style.display = 'none';
            selectedProvider = null;
            void loadSettings();
        }
        catch (err) {
            showToast('Failed to remove: ' + err.message, 'error');
        }
    }, { danger: true, confirmText: 'Remove' });
}
function renderMemorySettings(settings) {
    const budgetSelect = document.getElementById('memoryBudget');
    budgetSelect.value = settings.budget ?? 'minimal';
    const learningToggle = document.getElementById('autoLearningToggle');
    learningToggle.checked = settings.auto_learning !== false;
    document.getElementById('activeProviderDisplay').textContent = settings.active_provider ?? 'ollama';
}
export async function updateMemoryBudget() {
    const budget = document.getElementById('memoryBudget').value;
    try {
        await apiFetch(API_BASE + '/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ budget }),
        });
        showToast('Memory budget updated', 'success');
    }
    catch (err) {
        showToast('Failed to update: ' + err.message, 'error');
    }
}
export async function toggleAutoLearning() {
    const enabled = document.getElementById('autoLearningToggle').checked;
    try {
        await apiFetch(API_BASE + '/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ auto_learning: enabled }),
        });
    }
    catch (err) {
        showToast('Failed to update: ' + err.message, 'error');
    }
}
