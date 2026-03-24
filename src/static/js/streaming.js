/* streaming.js - LLM streaming functionality (single + dual) */

// -----------------------------
// Single-model streaming (existing)
// -----------------------------
async function streamExplanation(latex, targetElementId) {
  /**
   * Stream LLM explanation with real-time text display
   *
   * @param {string} latex - LaTeX expression to explain
   * @param {string} targetElementId - ID of element to display text in
   */
  const targetElement = document.getElementById(targetElementId);
  if (!targetElement) {
    console.error(`streamExplanation: target element "${targetElementId}" not found`);
    return;
  }

  targetElement.innerHTML = '<span class="typing-cursor"></span>';

  try {
    const response = await fetch('/api/explain-latex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latex: latex }),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let fullText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        try {
          const data = JSON.parse(line.substring(6));

          if (data.chunk) {
            fullText += data.chunk;
            const formattedText = formatMarkdown(fullText);
            targetElement.innerHTML = formattedText + '<span class="typing-cursor"></span>';
            targetElement.scrollTop = targetElement.scrollHeight;
          }

          if (data.done) {
            targetElement.innerHTML = formatMarkdown(fullText);

            // Re-render MathJax after streaming completes
            await safeTypesetMathJax([targetElement]);
            break;
          }

          if (data.error) {
            targetElement.innerHTML = `<p style="color: red;">Error: ${escapeHtml(data.error)}</p>`;
            break;
          }
        } catch (e) {
          console.error('Error parsing SSE data:', e);
        }
      }
    }
  } catch (error) {
    targetElement.innerHTML = `<p style="color: red;">Error: ${escapeHtml(error.message)}</p>`;
    console.error('Streaming error:', error);
  }
}

// -----------------------------
// Dual-model streaming (NEW)
// -----------------------------
async function streamExplanationDual(latex, modelKey, targetElementId) {
  /**
   * Stream explanation from a specific model (qwen / deepseek) into a target element.
   *
   * Expects your backend to accept: { latex, model }
   *
   * @param {string} latex
   * @param {string} modelKey - e.g. "qwen" or "deepseek"
   * @param {string} targetElementId
   */
  const targetElement = document.getElementById(targetElementId);
  if (!targetElement) {
    console.error(`streamExplanationDual: target element "${targetElementId}" not found`);
    return;
  }

  targetElement.innerHTML = '<span class="typing-cursor"></span>';

  try {
    const response = await fetch('/api/explain-latex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ latex: latex, model: modelKey }),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let fullText = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        try {
          const data = JSON.parse(line.substring(6));

          if (data.chunk) {
            fullText += data.chunk;
            const formattedText = formatMarkdown(fullText);
            targetElement.innerHTML = formattedText + '<span class="typing-cursor"></span>';
            targetElement.scrollTop = targetElement.scrollHeight;
          }

          if (data.done) {
            targetElement.innerHTML = formatMarkdown(fullText);

            // Re-render MathJax after streaming completes
            await safeTypesetMathJax([targetElement]);
            break;
          }

          if (data.error) {
            targetElement.innerHTML = `<p style="color: red;">Error: ${escapeHtml(data.error)}</p>`;
            break;
          }
        } catch (e) {
          console.error('Error parsing SSE data:', e);
        }
      }
    }
  } catch (error) {
    targetElement.innerHTML = `<p style="color: red;">Error: ${escapeHtml(error.message)}</p>`;
    console.error('Dual streaming error:', error);
  }
}

// -----------------------------
// Helpers
// -----------------------------
async function safeTypesetMathJax(elements = []) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    try {
      await window.MathJax.typesetPromise(elements);
    } catch (err) {
      console.error('MathJax rendering error:', err);
    }
  }
}

/**
 * Simple markdown-to-HTML converter for LLM output
 */
function formatMarkdown(text) {
  // Escape HTML first
  text = String(text).replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Convert **bold** to <strong>
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Convert *italic* to <em>
  text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Convert `code/italic` to <em> (for variables like `i`, `n`)
  text = text.replace(/`(.*?)`/g, '<em>$1</em>');

  // Convert single newlines to <br>
  text = text.replace(/\n/g, '<br>');

  // Convert bullet points to <li>
  text = text.replace(/\* (.+?)<br>/g, '<li>$1</li>');
  text = text.replace(/- (.+?)<br>/g, '<li>$1</li>');
  text = text.replace(/• (.+?)<br>/g, '<li>$1</li>');

  // Wrap consecutive <li> in <ul>
  text = text.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');

  // Convert double <br> to paragraph breaks
  text = text.replace(/<br><br>/g, '</p><p>');

  // Wrap in paragraph if needed
  if (!text.startsWith('<')) text = '<p>' + text + '</p>';

  // Cleanup
  text = text.replace(/<br><\/p>/g, '</p>');
  text = text.replace(/<p><br>/g, '<p>');

  return text;
}

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

// -----------------------------
// Export to window (so upload.js can call it)
// -----------------------------
window.streamExplanation = streamExplanation;
window.streamExplanationDual = streamExplanationDual;
window.formatMarkdown = formatMarkdown;