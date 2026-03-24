/* tutor.js - LaTeX Tutor with DUAL LLM Comparison for Tab 3 */

const latexInputTutor = document.getElementById('latex-input-tutor');
const previewTutor = document.getElementById('latex-preview-tutor');

// Live preview
latexInputTutor.addEventListener('input', () => {
    const latex = latexInputTutor.value;
    if (latex) {
        previewTutor.innerHTML = `$$${latex}$$`;
        if (window.MathJax) {
            MathJax.typesetPromise([previewTutor]);
        }
    } else {
        previewTutor.textContent = 'Type LaTeX to see preview...';
    }
});

async function explainLatexTutor() {
    const latex = latexInputTutor.value.trim();
    
    if (!latex) {
        alert('Please enter a LaTeX expression first!');
        return;
    }
    
    // Show dual explanation section
    document.getElementById('explanation-tutor').style.display = 'block';
    
    // Clear previous content
    document.getElementById('explanation-content-llm-a').innerHTML = '<span class="typing-cursor"></span>';
    document.getElementById('explanation-content-llm-b').innerHTML = '<span class="typing-cursor"></span>';
    
    // Stream both explanations in parallel
    await Promise.all([
        streamExplanationDual(latex, 'qwen', 'explanation-content-llm-a'),
        streamExplanationDual(latex, 'deepseek', 'explanation-content-llm-b')
    ]);
    
    // NO FEEDBACK - You'll use Microsoft Forms instead
}

async function streamExplanationDual(latex, modelKey, targetElementId) {
    const targetElement = document.getElementById(targetElementId);
    targetElement.innerHTML = '<span class="typing-cursor"></span>';
    
    try {
        const response = await fetch('/api/explain-latex', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latex: latex, model: modelKey })
        });
        
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        
        while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
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
                            
                            // CRITICAL: Re-render MathJax after streaming completes
                            if (window.MathJax && window.MathJax.typesetPromise) {
                                try {
                                    await window.MathJax.typesetPromise([targetElement]);
                                } catch (err) {
                                    console.error('MathJax rendering error:', err);
                                }
                            }
                            break;
                        }
                        
                        if (data.error) {
                            targetElement.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                            break;
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                }
            }
        }
    } catch (error) {
        targetElement.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
}

function setExample(latex) {
    latexInputTutor.value = latex;
    latexInputTutor.dispatchEvent(new Event('input'));
}