const analyzeButton = document.getElementById('analyze-button');
const reviewInput = document.getElementById('review');
const resultDiv = document.getElementById('result');

analyzeButton.addEventListener('click', () => {
    const review = reviewInput.value;

    // Fetch sentiment from API
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ review })
    })
    .then(response => response.json())
    .then(data => {
        const sentiment = data.sentiment;
        const confidenceScore = data.confidence_score;

        resultDiv.textContent = `Sentiment: ${sentiment} (Confidence Score: ${confidenceScore.toFixed(2)})`;
    })
    .catch(error => {
        console.error(error);
    });
});
