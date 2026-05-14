/**
 * Setup wizard — first-run onboarding flow (5 steps).
 */
import { API_BASE, apiFetch } from '../api.js';
const WIZARD_STEPS = 5;
let wizardProvider = 'ollama';
export function getWizardProvider() {
    return wizardProvider;
}
export function setWizardProvider(name) {
    wizardProvider = name;
}
export function onboardNext(step) {
    for (let i = 1; i <= WIZARD_STEPS; i++) {
        document.getElementById('onboardStep' + i).style.display = 'none';
    }
    document.getElementById('onboardStep' + step).style.display = 'block';
    if (step === 2) {
        const needsKey = wizardProvider !== 'ollama' && wizardProvider !== 'local_openai';
        document.getElementById('wizardKeySection').style.display = needsKey ? '' : 'none';
        document.getElementById('wizardLocalSection').style.display = needsKey ? 'none' : '';
        const title = needsKey
            ? 'Enter ' + wizardProvider.charAt(0).toUpperCase() + wizardProvider.slice(1) + ' API Key'
            : 'Confirm Local Setup';
        document.getElementById('wizardStep2Title').textContent = title;
    }
    if (step === 3) {
        void wizardSaveProvider();
    }
}
async function wizardSaveProvider() {
    const apiKey = document.getElementById('wizardApiKey').value.trim();
    const body = { name: wizardProvider, api_key: apiKey, enabled: true };
    try {
        await apiFetch(API_BASE + '/settings/providers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        await apiFetch(API_BASE + '/settings/active-provider', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: wizardProvider }),
        });
    }
    catch (_) {
        // Non-fatal — user can configure later in Settings
    }
}
export async function wizardTestConnection() {
    const resultDiv = document.getElementById('wizardTestResult');
    resultDiv.innerHTML = '<span class="text-muted">Testing connection...</span>';
    try {
        const r = await apiFetch(API_BASE + '/settings/providers/' + wizardProvider + '/test', {
            method: 'POST',
        });
        const data = await r.json();
        if (data.ok) {
            resultDiv.innerHTML = '<span style="color:#4ade80;">Connected! Found ' + data.models.length + ' models.</span>';
            const selectDiv = document.getElementById('wizardModelSelect');
            const dropdown = document.getElementById('wizardModelDropdown');
            dropdown.innerHTML = '<option value="">Select model...</option>';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                dropdown.appendChild(opt);
            });
            selectDiv.style.display = '';
        }
        else {
            resultDiv.textContent = '';
            const span = document.createElement('span');
            span.style.color = '#f87171';
            span.textContent = data.error ?? 'Connection failed';
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
export function closeOnboarding() {
    const modelDropdown = document.getElementById('wizardModelDropdown');
    if (modelDropdown?.value) {
        apiFetch(API_BASE + '/settings/providers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: wizardProvider,
                default_model: modelDropdown.value,
                enabled: true,
            }),
        }).catch(() => { });
    }
    document.getElementById('onboardingModal').style.display = 'none';
    localStorage.setItem('velqua_onboarding_done', 'true');
}
export function showOnboarding() {
    document.getElementById('onboardingModal').style.display = 'flex';
    onboardNext(1);
}
