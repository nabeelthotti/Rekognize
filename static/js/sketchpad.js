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

function startDrawing(event) {
    drawing = true;
    draw(event);
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

predictButton.addEventListener('click', () => {
    // Create a new canvas to draw a resized version of the drawing
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = 28;
    offscreenCanvas.height = 28;
    const offCtx = offscreenCanvas.getContext('2d');
    offCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
    
    const dataURL = offscreenCanvas.toDataURL('image/png');
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL.split(',')[1] })
    })
    .then(response => response.json())
    .then(data => {
        result.textContent = `Predicted Digit: ${data.digit}`;
    })
    .catch(error => {
        console.error('Error:', error);
        result.textContent = 'Error predicting digit.';
    });
});

clearButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    result.textContent = '';
});
