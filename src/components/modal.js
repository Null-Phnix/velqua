/**
 * Modal system — replaces native alert/confirm/prompt.
 * Also provides toast notifications and a spinner helper.
 */
export function showModal({ title, message, input, inputDefault, confirmText, confirmClass, onConfirm }) {
    const overlay = document.getElementById('velquaModal');
    const titleEl = document.getElementById('modalTitle');
    const msgEl = document.getElementById('modalMessage');
    const inputWrap = document.getElementById('modalInputWrap');
    const inputEl = document.getElementById('modalInput');
    const cancelBtn = document.getElementById('modalCancel');
    const confirmBtn = document.getElementById('modalConfirm');
    titleEl.textContent = title ?? 'Confirm';
    msgEl.textContent = message ?? '';
    confirmBtn.textContent = confirmText ?? 'Confirm';
    confirmBtn.className = 'modal-confirm-btn' + (confirmClass ? ' ' + confirmClass : '');
    if (input) {
        inputWrap.style.display = 'block';
        inputEl.value = inputDefault ?? '';
        setTimeout(() => { inputEl.focus(); inputEl.select(); }, 50);
    }
    else {
        inputWrap.style.display = 'none';
    }
    overlay.style.display = 'flex';
    function cleanup() {
        overlay.style.display = 'none';
        cancelBtn.onclick = null;
        confirmBtn.onclick = null;
        overlay.onclick = null;
        inputEl.onkeydown = null;
    }
    cancelBtn.onclick = cleanup;
    overlay.onclick = (e) => { if (e.target === overlay)
        cleanup(); };
    confirmBtn.onclick = () => {
        const value = input ? inputEl.value : true;
        cleanup();
        if (onConfirm)
            onConfirm(value);
    };
    if (input) {
        inputEl.onkeydown = (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                confirmBtn.click();
            }
        };
    }
}
export function showConfirm(message, onConfirm, opts) {
    showModal({
        title: opts?.title ?? 'Confirm',
        message,
        confirmText: opts?.confirmText ?? 'Confirm',
        confirmClass: opts?.danger ? 'danger' : '',
        onConfirm: () => onConfirm(),
    });
}
export function showPrompt(message, defaultValue, onSubmit) {
    showModal({
        title: 'Edit',
        message,
        input: true,
        inputDefault: defaultValue,
        confirmText: 'Save',
        onConfirm: (value) => {
            const str = value;
            if (str !== null && str.trim())
                onSubmit(str.trim());
        },
    });
}
export function showToast(message, type) {
    const container = document.getElementById('toastContainer');
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
export function createSpinner(text) {
    const div = document.createElement('div');
    div.className = 'loading-state';
    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    div.appendChild(spinner);
    div.appendChild(document.createTextNode(' ' + text));
    return div;
}
