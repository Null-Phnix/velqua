/**
 * License management — activation, status display, deactivation.
 */

import { API_BASE, apiFetch } from '../api.js';
import { showConfirm, showToast } from './modal.js';
import type { LicenseStatusData, LicenseActivateResponse } from '../types.js';

export async function loadLicenseStatus(): Promise<void> {
    try {
        const r = await apiFetch(API_BASE + '/license/status');
        const data = await r.json() as LicenseStatusData;
        renderLicenseStatus(data);
    } catch (_) {
        const el = document.getElementById('licenseStatus');
        if (el) el.innerHTML = '<span class="text-muted">Could not check license</span>';
    }
}

function renderLicenseStatus(data: LicenseStatusData): void {
    const el = document.getElementById('licenseStatus');
    if (!el) return;

    const deactivateBtn = document.getElementById('deactivateLicenseBtn') as HTMLButtonElement | null;
    const activateBtn = document.getElementById('activateLicenseBtn') as HTMLButtonElement | null;

    if (data.status === 'active') {
        el.innerHTML = '';
        const badge = document.createElement('span');
        badge.className = 'license-badge license-active';
        badge.textContent = 'Active';
        el.appendChild(badge);
        if (data.customer_email) {
            const email = document.createElement('span');
            email.className = 'text-muted text-sm';
            email.textContent = ' — ' + data.customer_email;
            el.appendChild(email);
        }
        if (deactivateBtn) deactivateBtn.style.display = '';
        if (activateBtn) activateBtn.textContent = 'Change License';
    } else if (data.status === 'trial') {
        el.innerHTML = '';
        const badge = document.createElement('span');
        badge.className = 'license-badge license-trial';
        badge.textContent = 'Trial';
        el.appendChild(badge);
        const msg = document.createElement('span');
        msg.className = 'text-muted text-sm';
        msg.textContent = ' — Activate a license to unlock the full version';
        el.appendChild(msg);
        if (deactivateBtn) deactivateBtn.style.display = 'none';
        if (activateBtn) activateBtn.textContent = 'Activate License';
    } else {
        el.innerHTML = '';
        const badge = document.createElement('span');
        badge.className = 'license-badge license-expired';
        badge.textContent = 'Expired';
        el.appendChild(badge);
        const msg = document.createElement('span');
        msg.className = 'text-muted text-sm';
        msg.textContent = ' — ' + (data.message ?? 'Please re-activate');
        el.appendChild(msg);
        if (deactivateBtn) deactivateBtn.style.display = '';
        if (activateBtn) activateBtn.textContent = 'Re-Activate';
    }
}

export function showLicenseModal(): void {
    const modal = document.getElementById('licenseModal') as HTMLElement;
    const input = document.getElementById('licenseKeyInput') as HTMLInputElement;
    const result = document.getElementById('licenseActivateResult') as HTMLElement;
    input.value = '';
    result.innerHTML = '';
    modal.style.display = 'flex';
    setTimeout(() => input.focus(), 50);
}

export function hideLicenseModal(): void {
    (document.getElementById('licenseModal') as HTMLElement).style.display = 'none';
}

export async function activateLicense(): Promise<void> {
    const key = (document.getElementById('licenseKeyInput') as HTMLInputElement).value.trim();
    if (!key) return;

    const result = document.getElementById('licenseActivateResult') as HTMLElement;
    result.innerHTML = '<span class="text-muted">Validating...</span>';

    try {
        const r = await apiFetch(API_BASE + '/license/activate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key }),
        });
        const data = await r.json() as LicenseActivateResponse;
        if (data.success) {
            result.innerHTML = '<span style="color:#4ade80;">License activated!</span>';
            setTimeout(() => {
                hideLicenseModal();
                void loadLicenseStatus();
                showToast('License activated', 'success');
            }, 1000);
        } else {
            result.textContent = '';
            const span = document.createElement('span');
            span.style.color = '#f87171';
            span.textContent = data.message ?? 'Activation failed';
            result.appendChild(span);
        }
    } catch (err) {
        result.textContent = '';
        const span = document.createElement('span');
        span.style.color = '#f87171';
        span.textContent = 'Error: ' + (err as Error).message;
        result.appendChild(span);
    }
}

export async function deactivateLicense(): Promise<void> {
    showConfirm('Deactivate your license? You can re-activate later.', async () => {
        try {
            await apiFetch(API_BASE + '/license/deactivate', { method: 'POST' });
            showToast('License deactivated', 'success');
            void loadLicenseStatus();
        } catch (err) {
            showToast('Failed: ' + (err as Error).message, 'error');
        }
    });
}
