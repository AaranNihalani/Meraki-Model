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

const codebookInput = document.getElementById('codebookInput');
const uploadCodebookBtn = document.getElementById('uploadCodebookBtn');
const codebookMsg = document.getElementById('codebookMsg');

const downloadResultsBtn = document.getElementById('downloadResultsBtn');
const codebookViewer = document.getElementById('codebookViewer');
const codebookTbody = document.getElementById('codebookTbody');
const viewCodebookBtn = document.getElementById('viewCodebookBtn');

let lastResults = null;

async function analyzeText() {
    const text = inputText.value.trim();
    if (!text) return;

    // Reset state
    setLoading(true);
    showError(null);
    resultsSection.classList.add('hidden');
    resultsList.innerHTML = '';
    downloadResultsBtn.classList.add('hidden');

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
                
                // Create tooltip content
                let explanationText = '';
                if (tag.explanation) {
                    const penaltyMsg = tag.explanation.sentiment_penalty > 0 
                        ? `<br><span class="penalty-text">⚠️ Penalty: -${(tag.explanation.sentiment_penalty * 100).toFixed(0)}% (Mismatch)</span>` 
                        : '';
                    explanationText = `
                        <div class="tag-explanation">
                            <strong>${tag.label}</strong><br>
                            Definition: ${tag.explanation.definition || 'N/A'}<br>
                            Alignment: ${tag.explanation.alignment_score ? (tag.explanation.alignment_score * 100).toFixed(1) + '%' : 'N/A'}
                            ${penaltyMsg}
                        </div>
                    `;
                }

                tagSpan.innerHTML = `${tag.label} <span class="tag-score">${Math.round(tag.score * 100)}%</span>${explanationText}`;
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
    downloadResultsBtn.classList.remove('hidden');
    
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

function downloadResults() {
    if (!lastResults || lastResults.length === 0) return;

    // Convert to CSV
    const headers = ['Sentence', 'Tags', 'Scores', 'Explanations'];
    const rows = lastResults.map(item => {
        const sentence = `"${item.sentence.replace(/"/g, '""')}"`;
        const tags = `"${item.tags.map(t => t.label).join('; ')}"`;
        const scores = `"${item.tags.map(t => Math.round(t.score * 100) + '%').join('; ')}"`;
        const explanations = `"${item.tags.map(t => (t.explanation && t.explanation.definition) ? t.explanation.definition : '').join('; ')}"`;
        return [sentence, tags, scores, explanations].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `meraki_results_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

let isCodebookVisible = false;

async function toggleCodebookView() {
    isCodebookVisible = !isCodebookVisible;
    
    if (isCodebookVisible) {
        viewCodebookBtn.textContent = 'Hide Codebook';
        codebookViewer.classList.remove('hidden');
        
        // Fetch and render codebook if empty
        if (codebookTbody.children.length === 0) {
            try {
                const resp = await fetch('/api/codebook/download');
                if (resp.ok) {
                    const data = await resp.json();
                    renderCodebookTable(data);
                }
            } catch (e) {
                console.error("Failed to load codebook", e);
            }
        }
    } else {
        viewCodebookBtn.textContent = 'View Codebook';
        codebookViewer.classList.add('hidden');
    }
}

function renderCodebookTable(data) {
    codebookTbody.innerHTML = '';
    // Sort keys alphabetically
    const keys = Object.keys(data).sort();
    
    keys.forEach(key => {
        const tr = document.createElement('tr');
        const tdLabel = document.createElement('td');
        tdLabel.textContent = key;
        tdLabel.style.fontWeight = '600';
        
        const tdDef = document.createElement('td');
        tdDef.textContent = data[key];
        
        tr.appendChild(tdLabel);
        tr.appendChild(tdDef);
        codebookTbody.appendChild(tr);
    });
}

function selectCodebookFile() {
    codebookInput.click();
}

async function uploadCodebook() {
    const file = codebookInput.files && codebookInput.files[0];
    if (!file) return;

    const form = new FormData();
    form.append('file', file);
    
    uploadCodebookBtn.disabled = true;
    uploadCodebookBtn.textContent = 'Uploading...';
    codebookMsg.className = 'hidden';

    try {
        const resp = await fetch('/api/codebook', { method: 'POST', body: form });
        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || 'Codebook upload failed');
        }
        const data = await resp.json();
        codebookMsg.textContent = `Success: ${data.message || 'Codebook updated'}`;
        codebookMsg.className = 'success-msg'; // You might need to add this class to CSS or reuse existing
        codebookMsg.style.color = 'var(--success)';
        codebookMsg.style.marginTop = '0.5rem';
        codebookMsg.classList.remove('hidden');
    } catch (e) {
        codebookMsg.textContent = `Error: ${e.message}`;
        codebookMsg.className = 'error-msg'; // You might need to add this class
        codebookMsg.style.color = 'var(--error)';
        codebookMsg.style.marginTop = '0.5rem';
        codebookMsg.classList.remove('hidden');
    } finally {
        uploadCodebookBtn.disabled = false;
        uploadCodebookBtn.textContent = 'Upload Codebook';
        codebookInput.value = ''; // Reset input
    }
}

codebookInput.addEventListener('change', () => {
    if (codebookInput.files && codebookInput.files[0]) {
        uploadCodebook();
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
        const tdExplanation = document.createElement('td');
        tdExplanation.className = 'explanation-cell';

        if (item.tags.length > 0) {
            tdTags.textContent = item.tags.map(t => `${t.label} (${Math.round(t.score * 100)}%)`).join(', ');
            
            // Format explanations
            const explanations = item.tags.map(t => {
                if (t.explanation) {
                    return `<strong>${t.label}</strong>: ${t.explanation.definition || ''} (Align: ${(t.explanation.alignment_score * 100).toFixed(0)}%)`;
                }
                return `<strong>${t.label}</strong>: No explanation`;
            });
            tdExplanation.innerHTML = explanations.join('<br>');
        } else {
            tdTags.textContent = '';
            tdExplanation.textContent = '';
        }
        
        tr.appendChild(tdSentence);
        tr.appendChild(tdTags);
        tr.appendChild(tdExplanation);
        tbody.appendChild(tr);
    });
    document.getElementById('resultsSection').classList.remove('hidden');
    table.classList.remove('hidden');
    table.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
