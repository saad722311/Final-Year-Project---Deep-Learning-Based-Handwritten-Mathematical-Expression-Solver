/* upload.js - Upload and test personalized model + confirm + streamed explanations */

let uploadedImageData = null;
let lastPredictedLatex = null;

// -----------------------------
// Upload / Preview
// -----------------------------
document.getElementById('imageUpload').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (!file) return;

  console.log('File selected:', file.name);

  const reader = new FileReader();
  reader.onload = function (event) {
    uploadedImageData = event.target.result;

    // Show preview
    const preview = document.getElementById('uploadedImagePreview');
    preview.src = uploadedImageData;
    preview.style.display = 'block';

    // Hide placeholder
    const placeholder = document.querySelector('.upload-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    // Show controls
    document.getElementById('uploadControls').style.display = 'flex';

    console.log('Image loaded and displayed');
  };
  reader.readAsDataURL(file);
});

function clearUpload() {
  uploadedImageData = null;
  lastPredictedLatex = null;

  document.getElementById('imageUpload').value = '';
  document.getElementById('uploadedImagePreview').style.display = 'none';

  const placeholder = document.querySelector('.upload-placeholder');
  if (placeholder) placeholder.style.display = 'flex';

  document.getElementById('uploadControls').style.display = 'none';
  document.getElementById('upload-results').style.display = 'none';

  // Reset confirm/correct/explain UI (if present)
  const correctSection = document.getElementById('uploadCorrectSection');
  if (correctSection) correctSection.style.display = 'none';

  const explainWrap = document.getElementById('upload-explanation');
  if (explainWrap) explainWrap.style.display = 'none';

  const llmA = document.getElementById('upload-explanation-llm-a');
  const llmB = document.getElementById('upload-explanation-llm-b');
  if (llmA) llmA.innerHTML = '';
  if (llmB) llmB.innerHTML = '';

  console.log('Upload cleared');
}

// -----------------------------
// Predict
// -----------------------------
async function testUploadedImage() {
  if (!uploadedImageData) {
    alert('Please upload an image first!');
    return;
  }

  console.log('🎯 Testing uploaded image with personalized model...');

  try {
    const response = await fetch('/api/test-uploaded', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: uploadedImageData }),
    });

    const data = await response.json();

    if (!data.success) {
      alert('Error: ' + data.error);
      return;
    }

    console.log('Prediction received:', data);

    // Show results section
    document.getElementById('upload-results').style.display = 'block';

    // Display processed image
    document.getElementById('processedImageDisplay').src = data.processed_image;

    // Display prediction (raw)
    document.getElementById('uploadPredCode').innerHTML =
      `<code>${escapeHtml(data.predicted || '')}</code>`;

    // Display prediction (rendered)
    const predDisplay = (data.predicted_display || data.predicted || '').trim();
    lastPredictedLatex = predDisplay;

    const renderEl = document.getElementById('uploadPredRender');
    renderEl.innerHTML = predDisplay ? `$$${predDisplay}$$` : '';

    console.log('Rendered LaTeX:', predDisplay);

    // Re-render MathJax (only this element)
    if (window.MathJax && window.MathJax.typesetPromise) {
      try {
        await window.MathJax.typesetPromise([renderEl]);
        console.log('MathJax rendering complete');
      } catch (err) {
        console.error('MathJax rendering error:', err);
      }
    }

    // Reset confirm/correct/explain UI each new prediction
    const correctSection = document.getElementById('uploadCorrectSection');
    if (correctSection) correctSection.style.display = 'none';

    const explainWrap = document.getElementById('upload-explanation');
    if (explainWrap) explainWrap.style.display = 'none';

    const llmA = document.getElementById('upload-explanation-llm-a');
    const llmB = document.getElementById('upload-explanation-llm-b');
    if (llmA) llmA.innerHTML = '';
    if (llmB) llmB.innerHTML = '';

    // Also reset correction input + preview if they exist
    const corrInput = document.getElementById('upload-latex-correct');
    if (corrInput) corrInput.value = predDisplay;

    updateUploadLatexPreview(predDisplay);

    // Scroll to results
    document.getElementById('upload-results').scrollIntoView({
      behavior: 'smooth',
      block: 'nearest',
    });
  } catch (error) {
    console.error('Error:', error);
    alert('Error testing image: ' + error.message);
  }
}

// -----------------------------
// Confirm / Correct / Explain (Streaming)
// Requires: tutor.js exports window.streamExplanationDual = streamExplanationDual;
// -----------------------------
function confirmUploadLatex(isYes) {
  if (!lastPredictedLatex) {
    alert('No prediction found yet. Please predict first.');
    return;
  }

  if (isYes) {
    startUploadStreamingExplanation(lastPredictedLatex);
  } else {
    const section = document.getElementById('uploadCorrectSection');
    if (!section) {
      alert('Correction UI not found. Did you add the uploadCorrectSection HTML?');
      return;
    }

    section.style.display = 'block';

    const input = document.getElementById('upload-latex-correct');
    if (input) {
      input.value = lastPredictedLatex;
      updateUploadLatexPreview(input.value);
      input.focus();
    }

    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function explainUploadLatexFromInput() {
  const input = document.getElementById('upload-latex-correct');
  const latex = (input?.value || '').trim();

  if (!latex) {
    alert('Please enter/correct the LaTeX first.');
    return;
  }

  startUploadStreamingExplanation(latex);
}

async function startUploadStreamingExplanation(latex) {
  const wrap = document.getElementById('upload-explanation');
  const llmA = document.getElementById('upload-explanation-llm-a');
  const llmB = document.getElementById('upload-explanation-llm-b');

  if (!wrap || !llmA || !llmB) {
    alert('Explanation UI not found. Did you add the upload-explanation HTML block?');
    return;
  }

  if (!window.streamExplanationDual) {
    alert('streamExplanationDual is not available. In tutor.js add: window.streamExplanationDual = streamExplanationDual;');
    return;
  }

  wrap.style.display = 'block';

  llmA.innerHTML = '<span class="typing-cursor"></span>';
  llmB.innerHTML = '<span class="typing-cursor"></span>';

  wrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  await Promise.all([
    window.streamExplanationDual(latex, 'qwen', 'upload-explanation-llm-a'),
    window.streamExplanationDual(latex, 'deepseek', 'upload-explanation-llm-b'),
  ]);
}

// -----------------------------
// Correction live preview
// -----------------------------
function updateUploadLatexPreview(latex) {
  const preview = document.getElementById('upload-latex-preview');
  if (!preview) return;

  const clean = (latex || '').trim();

  if (clean) {
    preview.innerHTML = `$$${clean}$$`;
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([preview]).catch((err) => {
        console.error('MathJax preview render error:', err);
      });
    }
  } else {
    preview.textContent = 'Type LaTeX to see preview...';
  }
}

// Attach live preview listener when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('upload-latex-correct');
  if (!input) return;

  input.addEventListener('input', () => {
    updateUploadLatexPreview(input.value);
  });
});

// -----------------------------
// Utils
// -----------------------------
function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}