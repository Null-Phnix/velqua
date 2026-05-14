/**
 * Timeline tab — chronological fact display.
 */
import { API_BASE, apiFetch } from '../api.js';
import { createSpinner } from './modal.js';
export async function loadTimeline() {
    const container = document.getElementById('timelineContainer');
    const statsDiv = document.getElementById('timelineStats');
    container.textContent = '';
    container.appendChild(createSpinner('Loading timeline...'));
    try {
        const response = await apiFetch(API_BASE + '/facts/timeline');
        const data = await response.json();
        statsDiv.textContent = data.total_facts + ' facts across ' + data.total_days + ' days';
        container.textContent = '';
        if (data.dates.length === 0) {
            const emptyP = document.createElement('p');
            emptyP.style.color = '#888';
            emptyP.textContent = 'No facts yet. Import some memories to see your timeline.';
            container.appendChild(emptyP);
            return;
        }
        data.dates.forEach(dateKey => {
            const group = data.groups[dateKey];
            const dayDiv = document.createElement('div');
            dayDiv.className = 'timeline-day';
            const dot = document.createElement('div');
            dot.className = 'timeline-dot';
            dayDiv.appendChild(dot);
            const dateDiv = document.createElement('div');
            dateDiv.className = 'timeline-date';
            const dateLabel = dateKey === 'unknown' ? 'Unknown date' :
                new Date(dateKey + 'T00:00:00').toLocaleDateString('en-US', {
                    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
                });
            dateDiv.textContent = dateLabel;
            const countBadge = document.createElement('span');
            countBadge.className = 'timeline-count';
            countBadge.textContent = group.length + ' fact' + (group.length !== 1 ? 's' : '');
            dateDiv.appendChild(countBadge);
            dayDiv.appendChild(dateDiv);
            group.forEach((fact) => {
                const factDiv = document.createElement('div');
                factDiv.className = 'timeline-fact';
                factDiv.textContent = fact.content;
                if (fact.topic || fact.category) {
                    const badges = document.createElement('div');
                    badges.className = 'pending-badges';
                    badges.style.marginTop = '4px';
                    if (fact.topic) {
                        const tb = document.createElement('span');
                        tb.className = 'badge badge-topic';
                        tb.textContent = fact.topic;
                        badges.appendChild(tb);
                    }
                    if (fact.category) {
                        const cb = document.createElement('span');
                        cb.className = 'badge badge-category';
                        cb.textContent = fact.category;
                        badges.appendChild(cb);
                    }
                    factDiv.appendChild(badges);
                }
                dayDiv.appendChild(factDiv);
            });
            container.appendChild(dayDiv);
        });
    }
    catch (error) {
        container.textContent = '';
        const errDiv = document.createElement('div');
        errDiv.className = 'error-boundary';
        errDiv.textContent = 'Failed to load timeline: ' + error.message;
        const retryBtn = document.createElement('button');
        retryBtn.textContent = 'Retry';
        retryBtn.addEventListener('click', loadTimeline);
        errDiv.appendChild(retryBtn);
        container.appendChild(errDiv);
    }
}
