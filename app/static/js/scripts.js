document.getElementById('image-upload').addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('radiograph').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

function predict() {
    const formData = new FormData(document.getElementById('prediction-form'));
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').value = data.result;
        document.getElementById('probability').value = data.probability + '%';
        document.getElementById('heatmap').src = data.heatmap_url;
    });
}

function resetForm() {
    document.getElementById('radiograph').src = '{{ url_for("static", filename="img/question_mark.svg") }}';
    document.getElementById('heatmap').src = '{{ url_for("static", filename="img/question_mark.svg") }}';
    document.getElementById('result').value = '';
    document.getElementById('probability').value = '';
}
