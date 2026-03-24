/* models.js - Model testing for 3 models: LSTM, Token-Aware, Transformer (Synthetic)
   New approach:
   - Ground truth preview stays unchanged.
   - Prediction render uses data.predicted_render:
       - if correct: backend sends ground_truth
       - if incorrect: backend sends predicted
*/

async function testAllModels() {
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    const btn = document.getElementById('testBtn');

    results.classList.remove('show');
    loading.classList.add('active');
    btn.disabled = true;

    try {
        console.log('🎯 Testing all 3 models...');

        console.log('1️⃣ Testing CNN-LSTM...');
        const lstmResponse = await fetch('/api/test-dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: 'lstm' })
        });

        const lstmData = await lstmResponse.json();
        if (!lstmData.success) throw new Error(lstmData.error || 'Failed to test CNN-LSTM');
        console.log(`✅ CNN-LSTM tested on: ${lstmData.filename}`);

        console.log('2️⃣ Testing Token-Aware...');
        const tokenAwareResponse = await fetch('/api/test-dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'token_aware',
                image_filename: lstmData.filename
            })
        });

        const tokenAwareData = await tokenAwareResponse.json();
        if (!tokenAwareData.success) throw new Error(tokenAwareData.error || 'Failed to test Token-Aware');
        console.log('✅ Token-Aware tested');

        console.log('3️⃣ Testing CNN-Transformer (Synthetic)...');
        const transformerResponse = await fetch('/api/test-dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'transformer_synthetic',
                image_filename: lstmData.filename
            })
        });

        const transformerData = await transformerResponse.json();
        if (!transformerData.success) throw new Error(transformerData.error || 'Failed to test CNN-Transformer');

        loading.classList.remove('active');
        btn.disabled = false;

        console.log('✅ All 3 models tested successfully!');
        displayResults(lstmData, tokenAwareData, transformerData);

    } catch (error) {
        loading.classList.remove('active');
        btn.disabled = false;
        alert('Error: ' + error.message);
        console.error('Full error:', error);
    }
}

function displayResults(lstmData, tokenAwareData, transformerData) {
    document.querySelector('.ground-truth-section').style.display = 'block';

    document.getElementById('datasetImage').src = lstmData.image;
    document.getElementById('filename').textContent = `File: ${lstmData.filename}`;

    // Ground truth raw text
    document.getElementById('groundTruthCode').innerHTML =
        `<code>${escapeHtml(lstmData.ground_truth)}</code>`;

    // ✅ KEEP GROUND TRUTH PREVIEW UNCHANGED
    document.getElementById('groundTruthRender').innerHTML =
        `$$${lstmData.ground_truth}$$`;

    displayModelResults('lstm', lstmData);
    displayModelResults('tokenAware', tokenAwareData);
    displayModelResults('transformer', transformerData);

    ['lstm', 'tokenAware', 'transformer'].forEach(model => {
        document.getElementById(`${model}Status`).style.display = 'block';
        document.getElementById(`${model}Sim`).parentElement.style.display = 'block';
        document.getElementById(`${model}Edit`).parentElement.style.display = 'block';
    });

    document.getElementById('results').classList.add('show');

    if (window.MathJax) {
        MathJax.typesetPromise();
    }
}

function displayModelResults(modelKey, data) {
    // Raw predicted code (always show what model actually output)
    document.getElementById(`${modelKey}PredCode`).innerHTML =
        `<code>${escapeHtml(data.predicted)}</code>`;

    // ✅ Render string is chosen by backend:
    // - correct => ground truth
    // - incorrect => predicted
    const renderLatex = data.predicted_render || data.predicted;

    // use innerHTML here (consistent with your GT render); MathJax will typeset later
    document.getElementById(`${modelKey}PredRender`).innerHTML =
        `$$${renderLatex}$$`;

    document.getElementById(`${modelKey}Sim`).textContent =
        data.similarity.toFixed(1) + '%';

    document.getElementById(`${modelKey}Edit`).textContent =
        data.edit_distance + ' chars';

    const status = document.getElementById(`${modelKey}Status`);
    if (data.correct) {
        status.className = 'status-indicator correct';
        status.textContent = '✅ CORRECT';
    } else {
        status.className = 'status-indicator incorrect';
        status.textContent = '❌ INCORRECT';
    }
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}