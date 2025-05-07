import cv2 as cv
import time
import numpy as np
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the embeddings and identities from the pickle file
embeddings_file = "face_embeddings.pkl"
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings, identities = pickle.load(f)
else:
    embeddings, identities = [], []

# Initialize face detection and recognition models
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def detect_and_align_faces(image):
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    image_pil = Image.fromarray(image_rgb)
    boxes, _ = mtcnn.detect(image_pil)
    faces, face_boxes = [], []
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
        return None
    face_pixels = cv.resize(face_pixels, (160, 160))
    face_pixels = (
        torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    face_pixels = (face_pixels - 127.5) / 128
    with torch.no_grad():
        embedding = model(face_pixels)
    return embedding.squeeze().cpu().numpy()


def recognize_face(embedding, embeddings, identities, threshold=0.4):
    min_dist = float("inf")
    best_match = None
    for i, saved_embedding in enumerate(embeddings):
        dist = cosine(embedding, saved_embedding)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            best_match = identities[i]
    return best_match if best_match else "Unknown", min_dist


def record_pictures(name, cap):
    training_path = "new_pictures"
    os.makedirs(training_path, exist_ok=True)
    patient_folder = os.path.join(training_path, name)
    os.makedirs(patient_folder, exist_ok=True)

    count = 0
    print("Recording will start in 2 seconds...")
    time.sleep(2)

    new_embeddings, new_identities = [], []

    while count <= 1:
        ret, frame = cap.read()
        if not ret:
            break
        faces, face_boxes = detect_and_align_faces(frame)
        for face, (x1, y1, x2, y2) in zip(faces, face_boxes):
            if face.size == 0:
                continue
            embedding = get_embedding(model, face)
            if embedding is not None:
                new_embeddings.append(embedding)
                new_identities.append(name)
                face_bgr = cv.cvtColor(face, cv.COLOR_RGB2BGR)
                file_name = os.path.join(patient_folder, f"{name}_{count:02d}.jpg")
                cv.imwrite(file_name, face_bgr)
                count += 1
        cv.imshow("Recording", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Update and save embeddings
    embeddings.extend(new_embeddings)
    identities.extend(new_identities)
    with open(embeddings_file, "wb") as f:
        pickle.dump((embeddings, identities), f)

    print(f"Recorded and saved embeddings for: {name}")


def recognize_from_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read image from: {image_path}")
        return

    faces, face_boxes = detect_and_align_faces(image)
    if not faces:
        print("No faces detected.")
        return

    for face, (x1, y1, x2, y2) in zip(faces, face_boxes):
        if face.size == 0:
            continue
        embedding = get_embedding(model, face)
        if embedding is None:
            continue
        label, distance = recognize_face(
            embedding, embeddings, identities, threshold=0.4
        )

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        text = label if label != "Unknown" else "Unknown"
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            image,
            text,
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        if label == "Unknown":
            cv.imshow("Result", image)
            print("Unknown face detected. Press 'A' to add or any other key to skip.")
            key = cv.waitKey(0)
            if key in [ord("a"), ord("A")]:
                name = input("Enter the person's name: ")
                cap = cv.VideoCapture(0)
                record_pictures(name, cap)
                cap.release()
                cv.destroyAllWindows()

    cv.imshow("Final Result", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


recognize_from_image("test_images/sample.jpg")
