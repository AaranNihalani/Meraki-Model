const API_URL = "/api/predict";

const inputText = document.getElementById('inputText');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const loader = document.getElementById('loader');
const errorMsg = document.getElementById('errorMsg');
const resultsSection = document.getElementById('resultsSection');
const resultsList = document.getElementById('resultsList');
const sentenceCount = document.getElementById('sentenceCount');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');

let lastResults = null;

async function analyzeText() {
    const text = inputText.value.trim();
    if (!text) return;

    // Reset state
    setLoading(true);
    showError(null);
    resultsSection.classList.add('hidden');
    resultsList.innerHTML = '';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Analysis failed. Please try again.');
        }

        const data = await response.json();
        lastResults = data.results;
        renderResults(lastResults);

    } catch (err) {
        showError(err.message);
    } finally {
        setLoading(false);
    }
}

function renderResults(results) {
    if (!results || results.length === 0) {
        showError("No sentences found to analyze.");
        return;
    }

    sentenceCount.textContent = `${results.length} Sentence${results.length !== 1 ? 's' : ''}`;

    results.forEach(item => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const sentenceDiv = document.createElement('div');
        sentenceDiv.className = 'sentence-text';
        sentenceDiv.textContent = item.sentence;

        const tagsDiv = document.createElement('div');
        tagsDiv.className = 'tags-row';

        if (item.tags.length > 0) {
            item.tags.forEach(tag => {
                const tagSpan = document.createElement('span');
                const isHighConf = tag.score >= 0.7;
                const isLowConf = tag.score < 0.7;
                tagSpan.className = `tag ${isHighConf ? 'high-conf' : ''} ${isLowConf ? 'low-conf' : ''}`;
                tagSpan.innerHTML = `${tag.label} <span class="tag-score">${Math.round(tag.score * 100)}%</span>`;
                tagsDiv.appendChild(tagSpan);
            });
        } else {
            const noTag = document.createElement('span');
            noTag.className = 'no-tags';
            noTag.textContent = 'No tags detected';
            tagsDiv.appendChild(noTag);
        }

        card.appendChild(sentenceDiv);
        card.appendChild(tagsDiv);
        resultsList.appendChild(card);
    });

    resultsSection.classList.remove('hidden');
    
    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
        loader.classList.remove('hidden');
        btnText.textContent = 'Analyzing...';
    } else {
        loader.classList.add('hidden');
        btnText.textContent = 'Analyze Text';
    }
}

function showError(msg) {
    if (msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    } else {
        errorMsg.classList.add('hidden');
    }
}

function clearText() {
    inputText.value = '';
    inputText.focus();
    resultsSection.classList.add('hidden');
    showError(null);
}

async function uploadFile() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
        showError('Select a file first.');
        return;
    }
    setLoading(true);
    showError(null);
    const form = new FormData();
    form.append('file', file);
    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: form });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || 'Upload failed');
        }
        const data = await resp.json();
        inputText.value = data.text || '';
        inputText.focus();
    } catch (e) {
        showError(e.message);
    } finally {
        setLoading(false);
    }
}

function selectFile() {
    fileInput.click();
}

fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
        uploadFile();
    }
});

function renderTable() {
    if (!lastResults || lastResults.length === 0) {
        showError('No results to show. Analyze text first.');
        return;
    }
    const tbody = document.getElementById('resultsTbody');
    const table = document.getElementById('resultsTable');
    tbody.innerHTML = '';
    lastResults.forEach(item => {
        const tr = document.createElement('tr');
        const tdSentence = document.createElement('td');
        tdSentence.textContent = item.sentence;
        const tdTags = document.createElement('td');
        if (item.tags.length > 0) {
            tdTags.textContent = item.tags.map(t => `${t.label} (${Math.round(t.score * 100)}%)`).join(', ');
        } else {
            tdTags.textContent = '';
        }
        tr.appendChild(tdSentence);
        tr.appendChild(tdTags);
        tbody.appendChild(tr);
    });
    document.getElementById('resultsSection').classList.remove('hidden');
    table.classList.remove('hidden');
    table.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
