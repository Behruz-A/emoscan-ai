import cv2
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

# AI model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "emotion-ferplus-8.onnx"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
input_name = input_info.name

EMOTIONS = [
    "NEUTRAL",
    "HAPPY",
    "SURPRISE",
    "SAD",
    "ANGRY",
    "DISGUST",
    "FEAR",
    "CONTEMPT"
]

DETECTION_INTERVAL = 4

emotion = "DETECTING..."
confidence = 0.0
last_detection = None


def detect_emotion(face):
    # Convert face to grayscale
    gray_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    # Resize to model input size
    gray_face = cv2.resize(
        gray_face,
        (64, 64)
    )

    # Prepare model input
    face_input = gray_face.astype(np.float32)
    face_input = face_input.reshape(1, 1, 64, 64)

    # Run AI model
    output = session.run(
        None,
        {input_name: face_input}
    )[0]

    scores = np.asarray(output).reshape(-1)

    # Convert scores to probabilities
    scores = scores - np.max(scores)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities)

    index = int(np.argmax(probabilities))

    return (
        EMOTIONS[index],
        float(probabilities[index] * 100)
    )


# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not found!")

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Face detector not loaded!")

print("EmotiVision AI started!")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(80, 80)
        )

        current_time = time.monotonic()

        if len(faces) > 0:

            # Select the largest face
            x, y, w, h = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Analyze every 4 seconds
            if (
                last_detection is None
                or current_time - last_detection >= DETECTION_INTERVAL
            ):
                face_image = frame[y:y+h, x:x+w].copy()

                try:
                    emotion, confidence = detect_emotion(
                        face_image
                    )

                    print(
                        f"Expression: {emotion} "
                        f"({confidence:.1f}%)"
                    )

                except Exception as e:
                    print("Detection error:", e)
                    emotion = "UNKNOWN"
                    confidence = 0.0

                last_detection = time.monotonic()

            # Show expression above face
            cv2.putText(
                frame,
                emotion,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:
            emotion = "NO FACE"
            confidence = 0.0
            last_detection = None

        # Display current result
        cv2.putText(
            frame,
            f"Expression: {emotion}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("EmotiVision AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("EmotiVision AI stopped.")
