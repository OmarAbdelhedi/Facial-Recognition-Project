import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

print(torch.cuda.get_device_name(0))  # This will print the name of the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
# Initialize InceptionResnetV1 for face embeddings
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def detect_and_align_faces(image):
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    image_pil = Image.fromarray(image_rgb)
    boxes, _ = mtcnn.detect(image_pil)
    faces = []
    if boxes is not None:
        # If multiple faces are detected, keep only the largest one
        if len(boxes) > 1:
            # Calculate box area and keep the largest
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            largest_face_idx = np.argmax(areas)
            boxes = [boxes[largest_face_idx]]  # Keep only the largest face

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image_rgb[y1:y2, x1:x2]
            faces.append(face)
    return faces


def get_embedding(model, face_pixels):
    if not isinstance(face_pixels, np.ndarray):
        face_pixels = np.array(face_pixels)
    face_pixels = cv.resize(face_pixels, (160, 160))
    face_pixels = (
        torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    face_pixels = (face_pixels - 127.5) / 128
    with torch.no_grad():
        embedding = model(face_pixels)
    return embedding.squeeze().cpu().numpy()


def load_embeddings(directory):
    embeddings = []
    labels = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                img = cv.imread(filepath)
                if img is not None:
                    detected_faces = detect_and_align_faces(img)
                    if detected_faces:
                        # Process only the first (or largest) face
                        face = detected_faces[0]
                        try:
                            embedding = get_embedding(model, face)
                            embeddings.append(embedding)
                            labels.append(name)
                        except:
                            continue
    return embeddings, labels


# takes around 12 minutes to load 15,072  pictures and returns 13,610 embedding
embeddings, labels = load_embeddings("train_img - Copy")
import pickle

with open("face_embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)

print("Embeddings and labels saved to face_embeddings.pkl")
