import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# ======================
# Configuration
# ======================
USER_NAME = "user5"  # change per user
SAVE_DIR = "embeddings"
OUTPUT_FILE = os.path.join(SAVE_DIR, f"{USER_NAME}_embedding.npy")
TOTAL_PER_POSE = 5
poses = ["front", "left", "right"]

os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize InsightFace (RetinaFace + ArcFace)
# app = FaceAnalysis(name="buffalo_l")  # detection + recognition
app = FaceAnalysis(name="buffalo_s")  # faster, smaller model
app.prepare(ctx_id=-1)  # CPU; use 0 for GPU

# Store all embeddings (across all poses)
all_embeddings = []

# ======================
# Countdown function
# ======================
def countdown(cap):
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(i), (260, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
        cv2.imshow("Capture", frame)
        cv2.waitKey(1000)


# ======================
# Get face embedding from frame
# ======================
def get_embedding_from_frame(frame):
    """Extract face embedding from frame using RetinaFace"""
    faces = app.get(frame)
    if len(faces) == 0:
        return None, None
    face = faces[0]
    return face.embedding, face.bbox  # 512-d vector, bounding box


# ======================
# Face Detection Preview
# ======================
def face_detection_preview(cap, pose="front"):
    """Show preview until user presses S"""
    # Set position message based on pose
    position_messages = {
        "front": "Position your face in the CENTER",
        "left": "Position your face to the LEFT side",
        "right": "Position your face to the RIGHT side"
    }
    position_msg = position_messages.get(pose, "Position your face in the center")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        faces = app.get(frame)
        if len(faces) > 0:
            x1, y1, x2, y2 = map(int, faces[0].bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, "Face Detected!", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, position_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press S to continue | Q to quit", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            return True
        elif key == ord("q"):
            return False


# ======================
# Initialize camera
# ======================
cap = cv2.VideoCapture(0)

# Show preview for initial positioning
ok = face_detection_preview(cap, "front")
if not ok:
    cap.release()
    cv2.destroyAllWindows()
    exit()

total_captured = 0

for pose in poses:
    ok = face_detection_preview(cap, pose)
    if not ok:
        break

    countdown(cap)

    saved = 0
    while saved < TOTAL_PER_POSE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        embedding, bbox = get_embedding_from_frame(frame)
        if embedding is None:
            cv2.putText(frame, "No face detected! Move to correct position.", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Capture", frame)
            if cv2.waitKey(400) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            continue

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.imshow("Capture", frame)

        # Store embedding
        all_embeddings.append(embedding)
        saved += 1
        total_captured += 1
        print(f"Captured {pose.upper()} embedding {saved}/{TOTAL_PER_POSE}")

        if cv2.waitKey(400) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

print("✅ Face embedding capture completed!")

# Average all embeddings into one vector
if len(all_embeddings) > 0:
    combined_embedding = np.mean(all_embeddings, axis=0)
    combined_embedding /= np.linalg.norm(combined_embedding)
    # avg_embedding = combined_embedding.astype(np.float16) cconvert into 16bit less memory space
    np.save(OUTPUT_FILE, combined_embedding)
    print(f"✅ Combined embedding saved to {OUTPUT_FILE}")
    print(f"Total embeddings captured: {len(all_embeddings)}")
    print(f"Embedding shape: {combined_embedding.shape}")

# Finish screen
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Capture Successful!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, "Press Q to Quit", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
