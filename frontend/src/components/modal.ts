/**
 * Modal system — replaces native alert/confirm/prompt.
 * Also provides toast notifications and a spinner helper.
 */

import type { ModalOptions, ShowConfirmOpts } from '../types.js';

export function showModal({ title, message, input, inputDefault, confirmText, confirmClass, onConfirm }: ModalOptions): void {
    const overlay = document.getElementById('velquaModal') as HTMLElement;
    const titleEl = document.getElementById('modalTitle') as HTMLElement;
    const msgEl = document.getElementById('modalMessage') as HTMLElement;
    const inputWrap = document.getElementById('modalInputWrap') as HTMLElement;
    const inputEl = document.getElementById('modalInput') as HTMLInputElement;
    const cancelBtn = document.getElementById('modalCancel') as HTMLButtonElement;
    const confirmBtn = document.getElementById('modalConfirm') as HTMLButtonElement;

    titleEl.textContent = title ?? 'Confirm';
    msgEl.textContent = message ?? '';
    confirmBtn.textContent = confirmText ?? 'Confirm';
    confirmBtn.className = 'modal-confirm-btn' + (confirmClass ? ' ' + confirmClass : '');

    if (input) {
        inputWrap.style.display = 'block';
        inputEl.value = inputDefault ?? '';
        setTimeout(() => { inputEl.focus(); inputEl.select(); }, 50);
    } else {
        inputWrap.style.display = 'none';
    }

    overlay.style.display = 'flex';

    function cleanup(): void {
        overlay.style.display = 'none';
        cancelBtn.onclick = null;
        confirmBtn.onclick = null;
        overlay.onclick = null;
        inputEl.onkeydown = null;
    }

    cancelBtn.onclick = cleanup;
    overlay.onclick = (e: MouseEvent) => { if (e.target === overlay) cleanup(); };

    confirmBtn.onclick = () => {
        const value: string | boolean = input ? inputEl.value : true;
        cleanup();
        if (onConfirm) onConfirm(value);
    };

    if (input) {
        inputEl.onkeydown = (e: KeyboardEvent) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                confirmBtn.click();
            }
        };
    }
}

export function showConfirm(message: string, onConfirm: () => void, opts?: ShowConfirmOpts): void {
    showModal({
        title: opts?.title ?? 'Confirm',
        message,
        confirmText: opts?.confirmText ?? 'Confirm',
        confirmClass: opts?.danger ? 'danger' : '',
        onConfirm: () => onConfirm(),
    });
}

export function showPrompt(message: string, defaultValue: string, onSubmit: (value: string) => void): void {
    showModal({
        title: 'Edit',
        message,
        input: true,
        inputDefault: defaultValue,
        confirmText: 'Save',
        onConfirm: (value) => {
            const str = value as string;
            if (str !== null && str.trim()) onSubmit(str.trim());
        },
    });
}

export function showToast(message: string, type?: string): void {
    const container = document.getElementById('toastContainer') as HTMLElement;
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + (type ?? 'info');
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

/** Creates a spinner element for loading states (avoids innerHTML). */
export function createSpinner(text: string): HTMLDivElement {
    const div = document.createElement('div');
    div.className = 'loading-state';
    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    div.appendChild(spinner);
    div.appendChild(document.createTextNode(' ' + text));
    return div;
}
