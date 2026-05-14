/**
 * Review queue — approve, reject, edit-approve pending facts.
 */
import { API_BASE, apiFetch } from '../api.js';
import { showConfirm, showPrompt, showToast, createSpinner } from './modal.js';
export async function loadPending() {
    const container = document.getElementById('pendingContainer');
    container.textContent = '';
    container.appendChild(createSpinner('Loading pending facts...'));
    try {
        const response = await apiFetch(API_BASE + '/review/pending');
        const data = await response.json();
        container.textContent = '';
        const badge = document.getElementById('reviewBadge');
        if (data.count > 0) {
            badge.textContent = String(data.count);
            badge.style.display = 'inline';
        }
        else {
            badge.style.display = 'none';
        }
        if (data.pending.length === 0) {
            const emptyP = document.createElement('p');
            emptyP.style.color = '#888';
            emptyP.textContent = 'No facts waiting for review. Facts will appear here as you chat through the proxy.';
            container.appendChild(emptyP);
            return;
        }
        data.pending.forEach((p) => {
            const item = document.createElement('div');
            item.className = 'pending-fact';
            const contentDiv = document.createElement('div');
            contentDiv.style.flex = '1';
            const contentText = document.createElement('span');
            contentText.textContent = p.content;
            contentDiv.appendChild(contentText);
            const qualityBadge = document.createElement('span');
            qualityBadge.className = 'quality-badge';
            qualityBadge.textContent = (p.quality_score * 100).toFixed(0) + '%';
            qualityBadge.style.marginLeft = '10px';
            contentDiv.appendChild(qualityBadge);
            const badges = document.createElement('div');
            badges.className = 'pending-badges';
            if (p.detected_topic) {
                const topicBadge = document.createElement('span');
                topicBadge.className = 'badge badge-topic';
                topicBadge.textContent = p.detected_topic;
                badges.appendChild(topicBadge);
            }
            if (p.detected_category) {
                const catBadge = document.createElement('span');
                catBadge.className = 'badge badge-category';
                catBadge.textContent = p.detected_category;
                badges.appendChild(catBadge);
            }
            if (p.detected_emotion) {
                const emotionBadge = document.createElement('span');
                emotionBadge.className = 'badge badge-emotion';
                emotionBadge.textContent = p.detected_emotion;
                badges.appendChild(emotionBadge);
            }
            if (badges.children.length > 0) {
                contentDiv.appendChild(badges);
            }
            if (p.contradictions && p.contradictions.length > 0) {
                p.contradictions.forEach(c => {
                    const warn = document.createElement('div');
                    warn.className = 'contradiction-warning';
                    warn.textContent = 'Contradicts: "' + c.content + '" (' + (c.confidence * 100).toFixed(0) + '% confidence)';
                    contentDiv.appendChild(warn);
                });
            }
            const actions = document.createElement('div');
            actions.className = 'pending-actions';
            const approveBtn = document.createElement('button');
            approveBtn.className = 'approve-btn';
            approveBtn.textContent = 'Approve';
            approveBtn.addEventListener('click', () => approvePending(p.id));
            const editApproveBtn = document.createElement('button');
            editApproveBtn.className = 'edit-approve-btn';
            editApproveBtn.textContent = 'Edit & Approve';
            editApproveBtn.addEventListener('click', () => editApprovePending(p.id, p.content));
            const rejectBtn = document.createElement('button');
            rejectBtn.className = 'reject-btn';
            rejectBtn.textContent = 'Reject';
            rejectBtn.addEventListener('click', () => rejectPending(p.id));
            actions.appendChild(approveBtn);
            actions.appendChild(editApproveBtn);
            actions.appendChild(rejectBtn);
            item.appendChild(contentDiv);
            item.appendChild(actions);
            container.appendChild(item);
        });
    }
    catch (error) {
        container.textContent = '';
        const errDiv = document.createElement('div');
        errDiv.className = 'error-boundary';
        errDiv.textContent = 'Failed to load review queue: ' + error.message;
        const retryBtn = document.createElement('button');
        retryBtn.textContent = 'Retry';
        retryBtn.addEventListener('click', loadPending);
        errDiv.appendChild(retryBtn);
        container.appendChild(errDiv);
    }
}
async function approvePending(id) {
    try {
        await apiFetch(API_BASE + '/review/approve/' + id, { method: 'POST' });
        void loadPending();
        showToast('Fact approved', 'success');
    }
    catch (error) {
        showToast('Approve failed: ' + error.message, 'error');
    }
}
async function rejectPending(id) {
    try {
        await apiFetch(API_BASE + '/review/reject/' + id, { method: 'POST' });
        void loadPending();
    }
    catch (error) {
        showToast('Reject failed: ' + error.message, 'error');
    }
}
export async function approveAllPending() {
    try {
        await apiFetch(API_BASE + '/review/approve-all', { method: 'POST' });
        void loadPending();
        showToast('All facts approved', 'success');
    }
    catch (error) {
        showToast('Approve all failed: ' + error.message, 'error');
    }
}
function editApprovePending(id, currentContent) {
    showPrompt('Edit fact content before approving:', currentContent, (newContent) => {
        apiFetch(API_BASE + '/review/edit-approve/' + id, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: newContent }),
        })
            .then(r => r.json())
            .then(() => { void loadPending(); showToast('Fact edited and approved', 'success'); })
            .catch((err) => showToast('Edit-approve failed: ' + err.message, 'error'));
    });
}
export async function rejectAllPending() {
    showConfirm('Reject all pending facts? This cannot be undone.', async () => {
        try {
            await apiFetch(API_BASE + '/review/reject-all', { method: 'POST' });
            void loadPending();
            showToast('All pending facts rejected', 'success');
        }
        catch (error) {
            showToast('Reject all failed: ' + error.message, 'error');
        }
    }, { title: 'Reject All', confirmText: 'Reject All', danger: true });
}
