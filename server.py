# this code is used to connect the front end to the back end
# not completed yet

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the embeddings and identities from the pickle file
embeddings_file = r"C:\Users\abdel\OneDrive\Bureau\test\embeddings_from_csv.pkl"
with open(embeddings_file, "rb") as f:
    embeddings, identities = pickle.load(f)


# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
# Initialize InceptionResnetV1 for face embeddings
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def recognize_face(embedding, embeddings, identities, threshold=0.4):
    min_dist = float("inf")
    best_match = None
    for i, saved_embedding in enumerate(embeddings):
        dist = cosine(embedding, saved_embedding)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            best_match = identities[i]
    if best_match is None:
        return "Unknown", min_dist
    return best_match, min_dist


def detect_and_align_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    image_pil = Image.fromarray(image_rgb)
    boxes, _ = mtcnn.detect(image_pil)
    faces = []
    face_boxes = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image_rgb[y1:y2, x1:x2]
            faces.append(face)
            face_boxes.append((x1, y1, x2, y2))
    return faces, face_boxes


def get_embedding(model, face_pixels):
    if not isinstance(face_pixels, np.ndarray):
        face_pixels = np.array(face_pixels)
    if face_pixels.size == 0:
        print("Warning: face_pixels is empty")
        return None
    face_pixels = cv2.resize(face_pixels, (160, 160))
    face_pixels = (
        torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    face_pixels = (face_pixels - 127.5) / 128
    with torch.no_grad():
        embedding = model(face_pixels)
    return embedding.squeeze().cpu().numpy()


app = Flask(__name__)


@app.route("/stream", methods=["POST"])
def stream():
    data = request.json
    image_data = data.get("image")

    if image_data:
        # Decode the base64 string into bytes
        img_bytes = base64.b64decode(image_data)

        # Convert bytes data to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode the numpy array into an image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            faces, face_boxes = detect_and_align_faces(img)
            for face, (x1, y1, x2, y2) in zip(faces, face_boxes):
                if face.size == 0:
                    continue
                embedding = get_embedding(model, face)
                if embedding is None:
                    continue
                label, distance = recognize_face(
                    embedding, embeddings, identities, threshold=0.4
                )

            return jsonify({"label": label})

    return jsonify({"label": "No valid image received"}), 400


if __name__ == "__main__":
    app.run(debug=True)
