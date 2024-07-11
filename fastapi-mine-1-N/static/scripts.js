document.addEventListener("DOMContentLoaded", () => {
    const openCameraButton = document.getElementById('open-camera');
    const takePhotoButton = document.getElementById('take-photo');
    const camera = document.getElementById('camera');
    const photoCanvas = document.getElementById('photo-canvas');
    const resultDiv = document.getElementById('result');
    const uploadedImg = document.getElementById('uploaded-img');
    const similarImg = document.getElementById('similar-img');
    let stream;

    document.getElementById('compare-face-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', document.getElementById('compare-face-file').files[0]);

        try {
            const response = await fetch('/compare_face/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred');
            }

            const result = await response.json();
            displayResult(result, URL.createObjectURL(document.getElementById('compare-face-file').files[0]));
        } catch (error) {
            console.error('Full error:', error);
            alert(`Error: ${error.message}`);
            resultDiv.textContent = `Error: ${error.message}`;
            uploadedImg.style.display = 'none';
            similarImg.style.display = 'none';
        }
    });

    openCameraButton.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            camera.srcObject = stream;
            camera.style.display = 'block';
            takePhotoButton.style.display = 'block';
        } catch (err) {
            console.error('Error accessing camera: ', err);
        }
    });

    takePhotoButton.addEventListener('click', () => {
        const context = photoCanvas.getContext('2d');
        photoCanvas.width = camera.videoWidth;
        photoCanvas.height = camera.videoHeight;
        context.drawImage(camera, 0, 0, photoCanvas.width, photoCanvas.height);
        
        photoCanvas.toBlob(blob => {
            const formData = new FormData();
            formData.append('file', blob, 'photo.jpg');

            fetch('/compare_face/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                displayResult(result, URL.createObjectURL(blob));
            })
            .catch(error => {
                console.error('Error:', error);
                resultDiv.textContent = 'An error occurred while comparing the face.';
            });
        }, 'image/jpeg');

        // Stop the camera stream
        stream.getTracks().forEach(track => track.stop());
        camera.style.display = 'none';
        takePhotoButton.style.display = 'none';
    });

    function displayResult(result, uploadedImgSrc) {
        resultDiv.textContent = `Similarity: ${result.similarity}`;
        uploadedImg.src = uploadedImgSrc;
        uploadedImg.style.display = 'block';

        if (result.most_similar_image) {
            similarImg.src = `/image/${encodeURIComponent(result.most_similar_image.split('/').pop())}`;
            similarImg.style.display = 'block';
        } else {
            similarImg.style.display = 'none';
            resultDiv.textContent += ' (No similar image found)';
        }
    }
});
