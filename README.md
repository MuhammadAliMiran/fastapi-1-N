# Face Similarity API with Qdrant Integration

This project provides a Face Similarity API built using FastAPI, dlib, OpenCV, PIL, and FaceNet. It integrates with Qdrant for storing and searching face embeddings.

## Features

- **Face Detection**: Detects faces in uploaded images using dlib's face detector.
- **Face Cropping and Padding**: Crops faces from the images with padding to ensure the faces are properly centered.
- **Face Embedding Extraction**: Uses FaceNet to extract face embeddings for similarity comparison.
- **Cosine Similarity Calculation**: Calculates the cosine similarity between the face embeddings of two images.
- **Qdrant Integration**: Stores and searches face embeddings in Qdrant for efficient similarity comparisons.
- **Web Interface**: Provides a simple web interface to upload images or take photos using the camera for comparison.

## Usage

### API Endpoints

* **Root Endpoint**: `GET /`
   * Returns a welcome message.

* **Get Image Endpoint**: `GET /image/{image_path:path}`
   * Parameters:
      * `image_path`: Path to the image.
   * Returns:
      * The requested image file.

* **Check Qdrant Endpoint**: `GET /check_qdrant`
   * Checks the status of the Qdrant collection and retrieves sample vectors.

* **Compare Face Endpoint**: `POST /compare_face/`
   * Parameters:
      * `file`: The image file to compare.
   * Returns:
      * `similarity`: The similarity score between the uploaded face and the most similar face in Qdrant.
      * `most_similar_image`: Path to the most similar image.

## Code Overview

### main.py

* **Imports and Initial Setup**: Imports necessary libraries and initializes the FastAPI app, logging, machine learning model (FaceNet), face detector (dlib), and Qdrant client.
* **Face Detection**: Uses dlib to detect faces in images.
* **Face Cropping and Padding**: Crops the detected face with padding to ensure proper centering.
* **Face Embedding Extraction**: Converts the face image to a format suitable for FaceNet and extracts embeddings.
* **Cosine Similarity Calculation**: Computes the cosine similarity between two face embeddings.
* **API Endpoints**: Implements the endpoints described in the Usage section.
* **Static Files**: Serves the web interface files from the `static` directory.

### index.html

* Provides a user interface to upload images or take photos for face comparison.

### scripts.js

* Handles the frontend logic for capturing images from the camera and submitting the form to the API.

### qdrant-storage.py

* **Qdrant Initialization**: Initializes the Qdrant client and FaceNet model.
* **Face Detection and Embedding Extraction**: Extracts faces from images and computes embeddings.
* **Store Embeddings**: Stores the computed face embeddings in the Qdrant collection for later retrieval and comparison.
