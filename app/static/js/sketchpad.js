const canvas = document.getElementById('sketchpad');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predict');
const clearButton = document.getElementById('clear');
const result = document.getElementById('result');

let drawing = false;

canvas.width = 280;  // Easier to draw on a larger canvas
canvas.height = 280;

// Initialize canvas with a white background
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseleave', stopDrawing);  // Stop drawing when cursor leaves the canvas

canvas.addEventListener('touchstart', startTouchDrawing);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchmove', drawTouch);
canvas.addEventListener('touchcancel', stopDrawing);

function startDrawing(event) {
    drawing = true;
    draw(event);
}

function startTouchDrawing(event) {
    event.preventDefault();
    drawing = true;
    drawTouch(event);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function draw(event) {
    if (!drawing) return;

    ctx.lineWidth = 15;  // Width of the stroke for visible drawing
    ctx.lineCap = 'round';  // Rounded ends of the drawn line
    ctx.strokeStyle = 'black';  // Draw with black

    const x = event.clientX - canvas.offsetLeft;
    const y = event.clientY - canvas.offsetTop;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function drawTouch(event) {
    if (!drawing) return;

    event.preventDefault();

    ctx.lineWidth = 15;  // Width of the stroke for visible drawing
    ctx.lineCap = 'round';  // Rounded ends of the drawn line
    ctx.strokeStyle = 'black';  // Draw with black

    const touch = event.touches[0];
    const x = touch.clientX - canvas.offsetLeft;
    const y = touch.clientY - canvas.offsetTop;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

predictButton.addEventListener('click', () => {
    const endpoint = window.location.pathname.includes('alphabet') ? '/predict_alphabet' : '/predict_digit';
    
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = 28;
    offscreenCanvas.height = 28;
    const offCtx = offscreenCanvas.getContext('2d');
    offCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

    const dataURL = offscreenCanvas.toDataURL('image/png');
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL.split(',')[1] })
    })
    .then(response => response.json())
    .then(data => {
        const confidence = (data.confidence * 100).toFixed(2);
        const color = confidence > 50 ? 'green' : 'red';
        const predictionText = window.location.pathname.includes('alphabet') ?
            `Prediction: ${data.letter}` :
            `Prediction: ${data.digit}`;
        const confidenceText = `<span style='color:${color};'>Confidence: ${confidence}%</span>`;
        
        result.innerHTML = `${predictionText}<br>${confidenceText}`;
    })
    .catch(error => {
        console.error('Error:', error);
        result.textContent = 'Error predicting.';
    });
});

clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    result.textContent = '';
});
