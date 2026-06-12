async function uploadImage() {

    const file =
        document.getElementById("image")
        .files[0];

    const formData = new FormData();

    formData.append("file", file);

    const response = await fetch(
        "http://localhost:8000/predict",
        {
            method: "POST",
            body: formData
        }
    );

    const data = await response.json();

    document.getElementById("result")
        .innerText =
        JSON.stringify(data, null, 2);
}