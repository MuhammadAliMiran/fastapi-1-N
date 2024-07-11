import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import cv2
import dlib
import torch
from facenet_pytorch import InceptionResnetV1

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load dlib's face detector
face_detector = dlib.get_frontal_face_detector()

def extract_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 1:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = image[y:y+h, x:x+w]
        return face_image
    else:
        return None

def get_face_embedding(face_image):
    face_image = cv2.resize(face_image, (160, 160))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = np.transpose(face_image, (2, 0, 1))
    face_image = torch.tensor(face_image).float().to(device)
    face_image = (face_image - 127.5) / 128.0
    face_image = face_image.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_image).cpu().numpy().flatten()
    return embedding

def store_embeddings(directory_path):
    client = QdrantClient("http://localhost:6333")  # Adjust URL/port if necessary
    
    # Check if collection exists, if not, create it
    if not client.get_collections().collections:
        client.create_collection(
            collection_name="face_embeddings",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    
    embeddings = []
    image_paths = []
    
    for i, filename in enumerate(os.listdir(directory_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            face_image = extract_face(image_path)
            if face_image is not None:
                embedding = get_face_embedding(face_image)
                embeddings.append(embedding)
                image_paths.append(image_path)
    
    # Prepare points in the correct format
    points = [
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"image_path": path}
        )
        for i, (embedding, path) in enumerate(zip(embeddings, image_paths))
    ]
    
    # Upload embeddings to Qdrant
    client.upsert(
        collection_name="face_embeddings",
        points=points
    )

    print(f"Stored {len(embeddings)} embeddings in Qdrant.")

# Example usage
if __name__ == "__main__":
    storage_directory = "/Users/mac/Documents/pics"  # Directory containing images
    store_embeddings(storage_directory)
