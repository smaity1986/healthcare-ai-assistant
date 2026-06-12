async function analyze() {

    const file =
        document.getElementById("image")
            .files[0];

    if (!file) {

        alert("Choose image");

        return;
    }

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    document.getElementById(
        "result"
    ).innerHTML =
        "Analyzing...";

    const response =
        await fetch(
            "http://localhost:8000/predict",
            {
                method: "POST",
                body: formData
            }
        );

    const data =
        await response.json();

    document.getElementById(
        "result"
    ).innerHTML =

        `
Prediction: ${data.prediction}

Confidence: ${data.confidence}%

Risk: ${data.risk}

Explanation:
${data.explanation}
`;
}