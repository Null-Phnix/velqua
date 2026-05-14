/**
 * Facts tab — list, search, edit, delete, merge, tags, detail view.
 */
import { API_BASE, apiFetch } from '../api.js';
import { showConfirm, showPrompt, showToast, createSpinner } from './modal.js';
let selectedFacts = new Set();
let currentFactsPage = 0;
const FACTS_PER_PAGE = 50;
let currentSearchQuery = '';
let searchDebounceTimer = null;
function updateSelectionButtons() {
    const bulkBtn = document.getElementById('bulkDeleteBtn');
    const mergeBtn = document.getElementById('mergeBtn');
    bulkBtn.style.display = selectedFacts.size > 0 ? 'inline-block' : 'none';
    mergeBtn.style.display = selectedFacts.size >= 2 ? 'inline-block' : 'none';
}
/** Highlight search terms in text using <mark> elements (XSS-safe via DOM API). */
function highlightText(text, query) {
    if (!query)
        return document.createTextNode(text);
    const fragment = document.createDocumentFragment();
    const lower = text.toLowerCase();
    const qLower = query.toLowerCase();
    let lastIndex = 0;
    let idx = lower.indexOf(qLower);
    while (idx !== -1) {
        if (idx > lastIndex) {
            fragment.appendChild(document.createTextNode(text.slice(lastIndex, idx)));
        }
        const mark = document.createElement('mark');
        mark.textContent = text.slice(idx, idx + query.length);
        fragment.appendChild(mark);
        lastIndex = idx + query.length;
        idx = lower.indexOf(qLower, lastIndex);
    }
    if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
    return fragment;
}
/** Build fact DOM elements. Uses createElement throughout to avoid innerHTML XSS. */
function renderFacts(facts, container) {
    container.textContent = '';
    if (facts.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.style.cssText = 'text-align:center; padding:32px 16px;';
        const icon = document.createElement('div');
        icon.style.cssText = 'font-size:2.5em; margin-bottom:12px; opacity:0.4;';
        icon.textContent = currentSearchQuery ? '🔍' : '🧠';
        const msg = document.createElement('p');
        msg.style.cssText = 'color:#888; margin:0 0 12px;';
        msg.textContent = currentSearchQuery
            ? 'No facts match "' + currentSearchQuery + '"'
            : 'No memories yet.';
        emptyDiv.appendChild(icon);
        emptyDiv.appendChild(msg);
        if (!currentSearchQuery) {
            const cta = document.createElement('p');
            cta.style.cssText = 'color:#667eea; font-size:0.9em; cursor:pointer;';
            cta.textContent = 'Import your AI chat history to get started →';
            cta.addEventListener('click', () => {
                document.querySelector('[data-tab="import"]')?.click();
            });
            emptyDiv.appendChild(cta);
        }
        container.appendChild(emptyDiv);
        return;
    }
    facts.forEach(f => {
        const factItem = document.createElement('div');
        factItem.className = 'fact-item';
        factItem.dataset['factId'] = f.id;
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'fact-checkbox';
        checkbox.checked = selectedFacts.has(f.id);
        checkbox.addEventListener('change', () => {
            if (checkbox.checked)
                selectedFacts.add(f.id);
            else
                selectedFacts.delete(f.id);
            updateSelectionButtons();
        });
        factItem.appendChild(checkbox);
        const contentDiv = document.createElement('div');
        contentDiv.className = 'fact-content';
        const contentP = document.createElement('p');
        contentP.className = 'fact-text';
        contentP.appendChild(highlightText(f.content, currentSearchQuery));
        contentDiv.appendChild(contentP);
        // Metadata badges
        const metaDiv = document.createElement('div');
        metaDiv.className = 'fact-meta';
        const typeBadge = document.createElement('span');
        typeBadge.className = 'fact-type-badge';
        typeBadge.textContent = f.type.replace('FactType.', '').replace('fact.', '');
        metaDiv.appendChild(typeBadge);
        const confBadge = document.createElement('span');
        confBadge.className = 'fact-confidence';
        confBadge.textContent = (f.confidence * 100).toFixed(0) + '%';
        metaDiv.appendChild(confBadge);
        if (f.topic) {
            const tb = document.createElement('span');
            tb.className = 'badge badge-topic';
            tb.textContent = f.topic;
            metaDiv.appendChild(tb);
        }
        if (f.category) {
            const cb = document.createElement('span');
            cb.className = 'badge badge-category';
            cb.textContent = f.category;
            metaDiv.appendChild(cb);
        }
        if (f.emotion) {
            const eb = document.createElement('span');
            eb.className = 'badge badge-emotion';
            eb.textContent = f.emotion;
            metaDiv.appendChild(eb);
        }
        contentDiv.appendChild(metaDiv);
        // Tags
        if (f.tags && f.tags.length > 0) {
            const tagsDiv = document.createElement('div');
            tagsDiv.className = 'fact-tags';
            f.tags.forEach(tag => {
                const chip = document.createElement('span');
                chip.className = 'tag-chip';
                chip.textContent = tag;
                const removeX = document.createElement('button');
                removeX.className = 'tag-remove';
                removeX.textContent = '×';
                removeX.title = 'Remove tag';
                removeX.addEventListener('click', (e) => {
                    e.stopPropagation();
                    removeTag(f.id, tag);
                });
                chip.appendChild(removeX);
                tagsDiv.appendChild(chip);
            });
            const addTagBtn = document.createElement('button');
            addTagBtn.className = 'tag-add-btn';
            addTagBtn.textContent = '+ tag';
            addTagBtn.addEventListener('click', () => promptAddTag(f.id));
            tagsDiv.appendChild(addTagBtn);
            contentDiv.appendChild(tagsDiv);
        }
        else {
            const addTagBtn = document.createElement('button');
            addTagBtn.className = 'tag-add-btn';
            addTagBtn.textContent = '+ tag';
            addTagBtn.addEventListener('click', () => promptAddTag(f.id));
            contentDiv.appendChild(addTagBtn);
        }
        factItem.appendChild(contentDiv);
        // Action buttons
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'fact-actions';
        const editBtn = document.createElement('button');
        editBtn.className = 'fact-edit-btn';
        editBtn.textContent = 'Edit';
        editBtn.addEventListener('click', () => editFact(f.id, f.content));
        actionsDiv.appendChild(editBtn);
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'fact-delete-btn';
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', () => deleteFact(f.id));
        actionsDiv.appendChild(deleteBtn);
        factItem.appendChild(actionsDiv);
        container.appendChild(factItem);
    });
}
export async function loadFacts() {
    const container = document.getElementById('factsContainer');
    container.textContent = '';
    container.appendChild(createSpinner('Loading facts...'));
    selectedFacts.clear();
    updateSelectionButtons();
    try {
        const offset = currentFactsPage * FACTS_PER_PAGE;
        const r = await apiFetch(API_BASE + `/facts/list?limit=${FACTS_PER_PAGE}&offset=${offset}`);
        const data = await r.json();
        container.textContent = '';
        renderFacts(data.facts, container);
        renderPagination(data.total, data.offset, data.limit);
        const statsBar = document.getElementById('factStatsBar');
        if (statsBar)
            statsBar.textContent = `${data.total.toLocaleString()} fact${data.total !== 1 ? 's' : ''} stored`;
    }
    catch (error) {
        container.textContent = '';
        const errDiv = document.createElement('div');
        errDiv.className = 'error-boundary';
        errDiv.textContent = 'Failed to load facts: ' + error.message;
        const retryBtn = document.createElement('button');
        retryBtn.textContent = 'Retry';
        retryBtn.addEventListener('click', loadFacts);
        errDiv.appendChild(retryBtn);
        container.appendChild(errDiv);
    }
}
function renderPagination(total, offset, limit) {
    const totalPages = Math.ceil(total / limit);
    const currentPage = Math.floor(offset / limit);
    const pageInfo = document.getElementById('pageInfo');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    if (pageInfo)
        pageInfo.textContent = `Page ${currentPage + 1} of ${Math.max(1, totalPages)} (${total} facts)`;
    if (prevBtn)
        prevBtn.disabled = currentPage === 0;
    if (nextBtn)
        nextBtn.disabled = currentPage >= totalPages - 1;
}
export async function searchFacts() {
    const searchInput = document.getElementById('factSearch');
    const q = searchInput.value.trim();
    currentSearchQuery = q;
    if (!q) {
        currentFactsPage = 0;
        return loadFacts();
    }
    const container = document.getElementById('factsContainer');
    container.textContent = '';
    container.appendChild(createSpinner('Searching...'));
    try {
        const r = await apiFetch(API_BASE + '/facts/search?q=' + encodeURIComponent(q));
        const data = await r.json();
        container.textContent = '';
        renderFacts(data.results, container);
    }
    catch (error) {
        container.textContent = '';
        const errDiv = document.createElement('div');
        errDiv.className = 'error-boundary';
        errDiv.textContent = 'Search failed: ' + error.message;
        container.appendChild(errDiv);
    }
}
async function deleteFact(factId) {
    showConfirm('Delete this fact?', async () => {
        try {
            await apiFetch(API_BASE + '/facts/' + factId, { method: 'DELETE' });
            selectedFacts.delete(factId);
            void loadFacts();
            showToast('Fact deleted', 'success');
        }
        catch (error) {
            showToast('Delete failed: ' + error.message, 'error');
        }
    }, { title: 'Delete Fact', confirmText: 'Delete', danger: true });
}
function editFact(factId, currentContent) {
    showPrompt('Edit fact:', currentContent, async (newContent) => {
        try {
            await apiFetch(API_BASE + '/facts/' + factId, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: newContent }),
            });
            void loadFacts();
            showToast('Fact updated', 'success');
        }
        catch (error) {
            showToast('Update failed: ' + error.message, 'error');
        }
    });
}
export async function bulkDeleteSelected() {
    if (selectedFacts.size === 0)
        return;
    showConfirm(`Delete ${selectedFacts.size} selected facts?`, async () => {
        try {
            await apiFetch(API_BASE + '/facts/bulk-delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fact_ids: [...selectedFacts] }),
            });
            selectedFacts.clear();
            void loadFacts();
            showToast('Facts deleted', 'success');
        }
        catch (error) {
            showToast('Bulk delete failed: ' + error.message, 'error');
        }
    }, { title: 'Bulk Delete', confirmText: 'Delete All', danger: true });
}
export async function mergeSelected() {
    if (selectedFacts.size < 2)
        return;
    showPrompt('Enter merged content:', '', async (mergedContent) => {
        try {
            await apiFetch(API_BASE + '/facts/merge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fact_ids: [...selectedFacts], merged_content: mergedContent }),
            });
            selectedFacts.clear();
            void loadFacts();
            showToast('Facts merged', 'success');
        }
        catch (error) {
            showToast('Merge failed: ' + error.message, 'error');
        }
    });
}
function promptAddTag(factId) {
    showPrompt('Enter tag name:', '', async (tag) => {
        try {
            await apiFetch(API_BASE + '/facts/' + factId + '/tags', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tags: [tag] }),
            });
            void loadFacts();
            showToast('Tag added', 'success');
        }
        catch (error) {
            showToast('Failed to add tag: ' + error.message, 'error');
        }
    });
}
async function removeTag(factId, tag) {
    try {
        await apiFetch(API_BASE + '/facts/' + factId + '/tags/' + encodeURIComponent(tag), {
            method: 'DELETE',
        });
        void loadFacts();
        showToast('Tag removed', 'success');
    }
    catch (error) {
        showToast('Failed to remove tag: ' + error.message, 'error');
    }
}
export async function loadFactTypes() {
    try {
        const r = await apiFetch(API_BASE + '/facts/types');
        const data = await r.json();
        const select = document.getElementById('factTypeFilter');
        if (!select)
            return;
        // Keep the "All types" option, add the rest
        while (select.options.length > 1)
            select.remove(1);
        data.types.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.value;
            opt.textContent = t.label;
            select.appendChild(opt);
        });
        select.addEventListener('change', () => {
            if (select.value) {
                void loadFactsByType(select.value);
            }
            else {
                void loadFacts();
            }
        });
    }
    catch (_) { }
}
async function loadFactsByType(factType) {
    const container = document.getElementById('factsContainer');
    container.textContent = '';
    container.appendChild(createSpinner('Filtering by type...'));
    try {
        const r = await apiFetch(API_BASE + '/facts/by-type/' + encodeURIComponent(factType));
        const data = await r.json();
        container.textContent = '';
        renderFacts(data.facts, container);
    }
    catch (_) {
        void loadFacts();
    }
}
/** Set up debounced search input and pagination buttons. */
export function init() {
    const searchInput = document.getElementById('factSearch');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            if (searchDebounceTimer)
                clearTimeout(searchDebounceTimer);
            searchDebounceTimer = setTimeout(() => {
                void searchFacts();
            }, 300);
        });
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (searchDebounceTimer)
                    clearTimeout(searchDebounceTimer);
                void searchFacts();
            }
        });
    }
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentFactsPage > 0) {
                currentFactsPage--;
                void loadFacts();
            }
        });
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            currentFactsPage++;
            void loadFacts();
        });
    }
}
