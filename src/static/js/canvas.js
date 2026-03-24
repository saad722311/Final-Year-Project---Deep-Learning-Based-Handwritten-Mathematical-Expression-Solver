/* canvas.js - Drawing canvas functionality for Tab 2 (Draw & Learn) */

const canvasLearn = document.getElementById('drawingCanvasLearn');
const ctxLearn = canvasLearn.getContext('2d');

let isDrawingLearn = false;
let strokesLearn = [];
let currentStrokeLearn = [];

// Canvas setup
ctxLearn.lineWidth = 5;
ctxLearn.lineCap = 'round';
ctxLearn.lineJoin = 'round';
ctxLearn.strokeStyle = '#000000';

// Drawing events
canvasLearn.addEventListener('mousedown', startDrawingLearn);
canvasLearn.addEventListener('mousemove', drawLearn);
canvasLearn.addEventListener('mouseup', stopDrawingLearn);
canvasLearn.addEventListener('mouseout', stopDrawingLearn);
canvasLearn.addEventListener('touchstart', handleTouchLearn);
canvasLearn.addEventListener('touchmove', handleTouchLearn);
canvasLearn.addEventListener('touchend', stopDrawingLearn);

function startDrawingLearn(e) {
    isDrawingLearn = true;
    currentStrokeLearn = [];
    const pos = getMousePosLearn(e);
    ctxLearn.beginPath();
    ctxLearn.moveTo(pos.x, pos.y);
    currentStrokeLearn.push({x: pos.x, y: pos.y});
}

function drawLearn(e) {
    if (!isDrawingLearn) return;
    const pos = getMousePosLearn(e);
    ctxLearn.lineTo(pos.x, pos.y);
    ctxLearn.stroke();
    currentStrokeLearn.push({x: pos.x, y: pos.y});
}

function stopDrawingLearn() {
    if (isDrawingLearn && currentStrokeLearn.length > 0) {
        strokesLearn.push([...currentStrokeLearn]);
        currentStrokeLearn = [];
    }
    isDrawingLearn = false;
}

function getMousePosLearn(e) {
    const rect = canvasLearn.getBoundingClientRect();
    const scaleX = canvasLearn.width / rect.width;
    const scaleY = canvasLearn.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function handleTouchLearn(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvasLearn.dispatchEvent(mouseEvent);
}

function clearCanvasLearn() {
    ctxLearn.clearRect(0, 0, canvasLearn.width, canvasLearn.height);
    strokesLearn = [];
    document.getElementById('predictions-learn').style.display = 'none';
    document.getElementById('explanation-learn').style.display = 'none';
}

function undoStrokeLearn() {
    if (strokesLearn.length === 0) return;
    strokesLearn.pop();
    redrawCanvasLearn();
}

function redrawCanvasLearn() {
    ctxLearn.clearRect(0, 0, canvasLearn.width, canvasLearn.height);
    strokesLearn.forEach(stroke => {
        if (stroke.length === 0) return;
        ctxLearn.beginPath();
        ctxLearn.moveTo(stroke[0].x, stroke[0].y);
        for (let i = 1; i < stroke.length; i++) {
            ctxLearn.lineTo(stroke[i].x, stroke[i].y);
        }
        ctxLearn.stroke();
    });
}

async function testDrawnLearn() {
    // Convert canvas to base64 with white background
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = canvasLearn.width;
    exportCanvas.height = canvasLearn.height;
    const exportCtx = exportCanvas.getContext('2d');
    
    exportCtx.fillStyle = '#FFFFFF';
    exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    exportCtx.drawImage(canvasLearn, 0, 0);
    
    const canvasDataURL = exportCanvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/api/test-drawn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: canvasDataURL })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Display predictions
            document.getElementById('lstm-pred-learn').textContent = data.results.lstm.predicted;
            document.getElementById('lstm-render-learn').innerHTML = `$$${data.results.lstm.predicted}$$`;
            
            document.getElementById('transformer-pred-learn').textContent = data.results.transformer.predicted;
            document.getElementById('transformer-render-learn').innerHTML = `$$${data.results.transformer.predicted}$$`;
            
            document.getElementById('tokenaware-pred-learn').textContent = data.results.token_aware.predicted;
            document.getElementById('tokenaware-render-learn').innerHTML = `$$${data.results.token_aware.predicted}$$`;
            
            document.getElementById('predictions-learn').style.display = 'block';
            
            // Re-render MathJax
            if (window.MathJax) {
                MathJax.typesetPromise();
            }
            
            // Enable confirmation when user selects
            document.querySelectorAll('input[name="correct-pred"]').forEach(radio => {
                radio.addEventListener('change', () => {
                    document.getElementById('explain-btn-learn').disabled = false;
                });
            });
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function explainExpression() {
    const selectedRadio = document.querySelector('input[name="correct-pred"]:checked');
    if (!selectedRadio) return;
    
    let latexToExplain;
    
    if (selectedRadio.value === 'custom') {
        latexToExplain = document.getElementById('custom-latex').value;
    } else if (selectedRadio.value === 'lstm') {
        latexToExplain = document.getElementById('lstm-pred-learn').textContent;
    } else if (selectedRadio.value === 'transformer') {
        latexToExplain = document.getElementById('transformer-pred-learn').textContent;
    } else if (selectedRadio.value === 'tokenaware') {
        latexToExplain = document.getElementById('tokenaware-pred-learn').textContent;
    }
    
    if (!latexToExplain || !latexToExplain.trim()) {
        alert('Please enter a LaTeX expression!');
        return;
    }
    
    // Show explanation section
    document.getElementById('explanation-learn').style.display = 'block';
    
    // Stream the explanation
    await streamExplanation(latexToExplain, 'explanation-content-learn');
}