import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# ==========================
# Config
# ==========================
EMBEDDINGS_DIR = "embeddings"
THRESHOLD = 0.4  # cosine distance threshold for recognition

# Load saved embeddings
user_embeddings = {}
for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".npy"):
        name = file.split("_")[0]  # e.g., "user1_embedding.npy"
        emb = np.load(os.path.join(EMBEDDINGS_DIR, file))
        user_embeddings[name] = emb

# Load ArcFace with RetinaFace for detection
app = FaceAnalysis(name="buffalo_l")  # detection + recognition
app.prepare(ctx_id=-1)  # CPU; use 0 for GPU

# Initialize webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces using RetinaFace
    faces = app.get(frame)  # returns list of face objects

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)  # bounding box
        embedding = face.embedding / np.linalg.norm(face.embedding)

        # Compare with stored embeddings
        best_match = "Unknown"
        best_score = 1.0
        for name, stored_emb in user_embeddings.items():
            score = cosine(embedding, stored_emb)
            if score < best_score:
                best_score = score
                best_match = name

        if best_score > THRESHOLD:
            best_match = "Unknown"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{best_match} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press Q to Quit",
            (35, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    # Show main frame
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
