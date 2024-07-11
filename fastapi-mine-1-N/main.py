from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
import numpy as np
import dlib
import cv2
from PIL import Image
import io
import torch
from facenet_pytorch import InceptionResnetV1
import logging
import os

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load dlib's face detector
face_detector = dlib.get_frontal_face_detector()

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# Check if Qdrant collection exists
try:
    collection_info = qdrant_client.get_collection("face_embeddings")
    logging.info(f"Qdrant collection 'face_embeddings' exists with {collection_info.vectors_count} vectors")
except Exception as e:
    logging.error(f"Error accessing Qdrant collection: {str(e)}")
    # Optionally, create the collection here if it doesn't exist
    
@app.get("/image/{image_path:path}")
async def get_image(image_path: str):
    full_path = os.path.join("/Users/mac/Documents/pics", image_path)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

def extract_face(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = np.array(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    
    padding_top = 100  # Define the amount of padding you want at the top
    padding_sides = 50  # Define the amount of padding you want on the sides and bottom

    if len(faces) == 1:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Calculate the coordinates with extra padding at the top and regular padding on other sides
        x_padded = max(x - padding_sides, 0)
        y_padded = max(y - padding_top, 0)
        w_padded = min(x + w + padding_sides, image.shape[1]) - x_padded
        h_padded = min(y + h + padding_sides, image.shape[0]) - y_padded
        
        # Extract the padded face region
        face_image = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        return face_image

    elif len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    else:
        raise HTTPException(status_code=400, detail="Multiple faces detected in the image")

def get_face_embedding(face_image):
    face_image = cv2.resize(face_image, (160, 160))
    face_image = np.transpose(face_image, (2, 0, 1))
    face_image = torch.tensor(face_image).float().to(device)
    face_image = (face_image - 127.5) / 128.0
    face_image = face_image.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_image).cpu().numpy().flatten()
    return embedding

@app.get("/check_qdrant")
async def check_qdrant():
    try:
        collection_info = qdrant_client.get_collection("face_embeddings")
        vectors = qdrant_client.scroll(
            collection_name="face_embeddings",
            limit=10  # Adjust as needed
        )
        return {
            "collection_info": collection_info.dict(),
            "sample_vectors": [v.dict() for v in vectors[0]]
        }
    except Exception as e:
        logging.error(f"Error checking Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking Qdrant")
    
@app.post("/compare_face/")
async def compare_face(file: UploadFile = File(...)):
    contents = await file.read()
    
    try:
        face_image = extract_face(contents)
        embedding = get_face_embedding(face_image)
        logging.info(f"Generated embedding of length: {len(embedding)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in face extraction or embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    try:
        search_result = qdrant_client.search(
            collection_name="face_embeddings",
            query_vector=embedding.tolist(),
            limit=1
        )
        
        logging.info(f"Search result: {search_result}")
        
        if search_result:
            most_similar = search_result[0]
            similarity = most_similar.score  # Cosine distance to similarity
            similar_image_path = most_similar.payload["image_path"]
            
            logging.info(f"Raw score (distance): {most_similar.score}")
            logging.info(f"Calculated similarity: {similarity}")
            logging.info(f"Similar image path: {similar_image_path}")
            
            return JSONResponse(content=jsonable_encoder({
                "similarity": round(similarity, 5),
                "most_similar_image": similar_image_path
            }))
        else:
            logging.warning("No similar faces found in Qdrant")
            return JSONResponse(content={"similarity": 0, "most_similar_image": None})
    except Exception as e:
        logging.error(f"Error in Qdrant search: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in face comparison")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)