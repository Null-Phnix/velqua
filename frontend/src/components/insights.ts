/**
 * Insights tab — analytics report, quality scores, graph stats, emotional patterns.
 */

import { API_BASE, apiFetch } from '../api.js';
import { createSpinner } from './modal.js';
import type { AnalyticsReport, QualityReport, GraphStats, EmotionalHistory } from '../types.js';

export async function loadInsights(): Promise<void> {
    const container = document.getElementById('insightsContainer') as HTMLElement;
    container.textContent = '';
    container.appendChild(createSpinner('Generating analytics report...'));

    try {
        const r = await apiFetch(API_BASE + '/analytics/report');
        const data = await r.json() as AnalyticsReport;
        container.textContent = '';

        if (data.error) {
            const errP = document.createElement('p');
            errP.style.color = '#888';
            errP.textContent = data.error;
            container.appendChild(errP);
            return;
        }

        // Memory Health cards
        const healthGrid = document.createElement('div');
        healthGrid.className = 'insights-grid';
        const healthItems = [
            { label: 'Healthy',   value: data.health.healthy,   cls: 'insight-healthy' },
            { label: 'Aging',     value: data.health.aging,     cls: 'insight-aging' },
            { label: 'At Risk',   value: data.health.at_risk,   cls: 'insight-atrisk' },
            { label: 'Forgotten', value: data.health.forgotten, cls: 'insight-forgotten' },
        ];
        healthItems.forEach(h => {
            const card = document.createElement('div');
            card.className = 'insight-card ' + h.cls;
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.textContent = String(h.value);
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = h.label;
            card.appendChild(val);
            card.appendChild(lbl);
            healthGrid.appendChild(card);
        });
        container.appendChild(healthGrid);

        // Summary stats
        const summaryGrid = document.createElement('div');
        summaryGrid.className = 'insights-grid';
        const summaryItems = [
            { label: 'Total Facts',    value: String(data.total_facts) },
            { label: 'Total Episodes', value: String(data.total_episodes) },
            { label: 'Memory Span',    value: data.memory_span_days + ' days' },
            { label: 'Topic Diversity', value: String(data.topic_diversity) },
        ];
        summaryItems.forEach(s => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.style.fontSize = '1.4em';
            val.textContent = s.value;
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = s.label;
            card.appendChild(val);
            card.appendChild(lbl);
            summaryGrid.appendChild(card);
        });
        container.appendChild(summaryGrid);

        // Top Topics bar chart
        if (data.top_topics?.length > 0) {
            const topicSection = document.createElement('div');
            topicSection.className = 'insight-section';
            const topicTitle = document.createElement('h4');
            topicTitle.textContent = 'Top Topics';
            topicSection.appendChild(topicTitle);

            const maxCount = Math.max(...data.top_topics.map(t => t.count));
            data.top_topics.forEach(t => {
                topicSection.appendChild(_makeBar(t.topic, t.count, maxCount));
            });
            container.appendChild(topicSection);
        }

        // Fact Quality summary
        const qualitySection = document.createElement('div');
        qualitySection.className = 'insight-section';
        const qualityTitle = document.createElement('h4');
        qualityTitle.textContent = 'Fact Quality';
        qualitySection.appendChild(qualityTitle);
        const qualityGrid = document.createElement('div');
        qualityGrid.className = 'insights-grid';
        [
            { label: 'Avg Confidence',    value: String(data.avg_fact_confidence) },
            { label: 'Avg Importance',    value: String(data.avg_episode_importance) },
            { label: 'Emotional Balance', value: String(data.emotional_balance) },
        ].forEach(q => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.style.fontSize = '1.4em';
            val.textContent = q.value;
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = q.label;
            card.appendChild(val);
            card.appendChild(lbl);
            qualityGrid.appendChild(card);
        });
        qualitySection.appendChild(qualityGrid);
        container.appendChild(qualitySection);

        // Facts by type
        if (data.facts_by_type && Object.keys(data.facts_by_type).length > 0) {
            const typeSection = document.createElement('div');
            typeSection.className = 'insight-section';
            const typeTitle = document.createElement('h4');
            typeTitle.textContent = 'Facts by Type';
            typeSection.appendChild(typeTitle);
            const maxTypeCount = Math.max(...Object.values(data.facts_by_type));
            Object.entries(data.facts_by_type).forEach(([type, count]) => {
                const label = type.replace('fact.', '').replace('FactType.', '');
                typeSection.appendChild(_makeBar(label, count, maxTypeCount));
            });
            container.appendChild(typeSection);
        }

        // Load additional insights in parallel
        void loadQualityInsights(container);
        void loadGraphInsights(container);
        void loadEmotionalInsights(container);

    } catch (error) {
        container.textContent = '';
        const errDiv = document.createElement('div');
        errDiv.className = 'error-boundary';
        errDiv.textContent = 'Failed to load insights: ' + (error as Error).message;
        const retryBtn = document.createElement('button');
        retryBtn.textContent = 'Retry';
        retryBtn.addEventListener('click', loadInsights);
        errDiv.appendChild(retryBtn);
        container.appendChild(errDiv);
    }
}

function _makeBar(label: string, count: number, maxCount: number): HTMLElement {
    const bar = document.createElement('div');
    bar.className = 'insight-bar';
    const labelEl = document.createElement('div');
    labelEl.className = 'insight-bar-label';
    labelEl.textContent = label;
    const track = document.createElement('div');
    track.className = 'insight-bar-track';
    const fill = document.createElement('div');
    fill.className = 'insight-bar-fill';
    fill.style.width = (maxCount > 0 ? (count / maxCount) * 100 : 0) + '%';
    track.appendChild(fill);
    const countEl = document.createElement('div');
    countEl.className = 'insight-bar-count';
    countEl.textContent = String(count);
    bar.appendChild(labelEl);
    bar.appendChild(track);
    bar.appendChild(countEl);
    return bar;
}

async function loadQualityInsights(container: HTMLElement): Promise<void> {
    try {
        const r = await apiFetch(API_BASE + '/analytics/quality');
        const data = await r.json() as QualityReport;
        if (data.error || !data.stats) return;

        const section = document.createElement('div');
        section.className = 'insight-section';
        const title = document.createElement('h4');
        title.textContent = 'Fact Quality Scores';
        section.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'insights-grid';
        [
            { label: 'Avg Quality',  value: (data.stats.avg_quality * 100).toFixed(0) + '%' },
            { label: 'Total Scored', value: String(data.stats.total) },
        ].forEach(s => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.style.fontSize = '1.4em';
            val.textContent = s.value;
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = s.label;
            card.appendChild(val);
            card.appendChild(lbl);
            grid.appendChild(card);
        });
        section.appendChild(grid);

        if (data.stats.distribution) {
            const dist = data.stats.distribution;
            const maxCount = Math.max(...Object.values(dist), 1);
            Object.entries(dist).forEach(([level, count]) => {
                section.appendChild(_makeBar(level, count, maxCount));
            });
        }

        if (data.stats.common_issues?.length > 0) {
            const issuesTitle = document.createElement('h5');
            issuesTitle.textContent = 'Common Issues';
            issuesTitle.style.color = '#fbbf24';
            issuesTitle.style.marginTop = '12px';
            section.appendChild(issuesTitle);
            data.stats.common_issues.forEach(issue => {
                const p = document.createElement('p');
                p.style.color = '#aaa';
                p.style.fontSize = '0.85em';
                p.style.margin = '4px 0';
                p.textContent = issue;
                section.appendChild(p);
            });
        }

        container.appendChild(section);
    } catch (_) {}
}

async function loadGraphInsights(container: HTMLElement): Promise<void> {
    try {
        const r = await apiFetch(API_BASE + '/graph/stats');
        const data = await r.json() as GraphStats;
        if (data.error) return;

        const section = document.createElement('div');
        section.className = 'insight-section';
        const title = document.createElement('h4');
        title.textContent = 'Memory Graph';
        section.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'insights-grid';
        const totalCard = document.createElement('div');
        totalCard.className = 'insight-card';
        const totalVal = document.createElement('div');
        totalVal.className = 'insight-value';
        totalVal.style.fontSize = '1.4em';
        totalVal.textContent = String(data.total_links ?? 0);
        const totalLbl = document.createElement('div');
        totalLbl.className = 'insight-label';
        totalLbl.textContent = 'Total Links';
        totalCard.appendChild(totalVal);
        totalCard.appendChild(totalLbl);
        grid.appendChild(totalCard);
        section.appendChild(grid);

        if (data.by_type && Object.keys(data.by_type).length > 0) {
            const maxCount = Math.max(...Object.values(data.by_type), 1);
            Object.entries(data.by_type).forEach(([type, count]) => {
                section.appendChild(_makeBar(type, count, maxCount));
            });
        }

        container.appendChild(section);
    } catch (_) {}
}

async function loadEmotionalInsights(container: HTMLElement): Promise<void> {
    try {
        const r = await apiFetch(API_BASE + '/retrieval/emotional/history?days=30');
        const data = await r.json() as EmotionalHistory;
        if (data.error) return;

        const section = document.createElement('div');
        section.className = 'insight-section';
        const title = document.createElement('h4');
        title.textContent = 'Emotional Patterns (30 days)';
        section.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'insights-grid';

        if (data.episode_count !== undefined) {
            const card = document.createElement('div');
            card.className = 'insight-card';
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.style.fontSize = '1.4em';
            val.textContent = String(data.episode_count);
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = 'Episodes Analyzed';
            card.appendChild(val);
            card.appendChild(lbl);
            grid.appendChild(card);
        }

        if (data.dominant_valence) {
            const card = document.createElement('div');
            card.className = 'insight-card';
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.style.fontSize = '1.2em';
            val.textContent = data.dominant_valence;
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = 'Dominant Mood';
            card.appendChild(val);
            card.appendChild(lbl);
            grid.appendChild(card);
        }

        if (data.avg_sentiment !== undefined) {
            const card = document.createElement('div');
            card.className = 'insight-card';
            const val = document.createElement('div');
            val.className = 'insight-value';
            val.style.fontSize = '1.4em';
            val.textContent = data.avg_sentiment.toFixed(2);
            const lbl = document.createElement('div');
            lbl.className = 'insight-label';
            lbl.textContent = 'Avg Sentiment';
            card.appendChild(val);
            card.appendChild(lbl);
            grid.appendChild(card);
        }

        if (grid.children.length > 0) section.appendChild(grid);

        if (data.valence_distribution && Object.keys(data.valence_distribution).length > 0) {
            const maxCount = Math.max(...Object.values(data.valence_distribution), 1);
            Object.entries(data.valence_distribution).forEach(([valence, count]) => {
                section.appendChild(_makeBar(valence, count, maxCount));
            });
        }

        if (section.children.length > 1) container.appendChild(section);
    } catch (_) {}
}
