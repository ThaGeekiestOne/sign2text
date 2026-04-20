import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

from preprocessing import preprocess, IMG_SIZE
from gestures import is_open_palm, are_both_palms_open


# Loading model
model = tf.keras.models.load_model("best_asl_eff.h5", compile=False)
dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model(dummy, training=False)
print("Model ready!")


# Lables
labels_dict = {
    0: '_', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E',
    6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K',
    12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q',
    18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
    24: 'X', 25: 'Y', 26: 'Z'
}


# mediapipe for hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.5,
    model_complexity=0
)


# basic setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
PREDICT_EVERY = 3

label = 0
confidence = 0.0
filtered = None
string = ""

last_capture_time = time.time()
CAPTURE_COOLDOWN = 3.0
stable_label = 0
stable_count = 0
STABLE_THRESHOLD = 6
MIN_CONFIDENCE = 60.0

# Gesture cooldown
last_space_time = time.time()
last_clear_time = time.time()
GESTURE_COOLDOWN = 2.0

# Gesture hold counters
open_palm_count = 0
PALM_HOLD = 20

print("Gestures: Open palm = SPACE | Two hands = DELETE | 'q' = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_count += 1

    small = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_found = False
    num_hands = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    now = time.time()

    # two hands plam open will clear last character
    if num_hands == 2 and are_both_palms_open(result.multi_hand_landmarks) and now - last_clear_time > GESTURE_COOLDOWN:
        if string:
            string = string[:-1]
        last_clear_time = now
        stable_count = 0
        print(f"Deleted last char | Word: '{string}'")

    # one hand palm open adds space
    elif num_hands == 1:
        hand_found = True
        hand_landmarks = result.multi_hand_landmarks[0]

        # ── Open palm check = SPACE ──
        if is_open_palm(hand_landmarks):
            open_palm_count += 1
            if open_palm_count >= PALM_HOLD and now - last_space_time > GESTURE_COOLDOWN:
                string += " "
                last_space_time = now
                open_palm_count = 0
                print(f"Space added | Word: '{string}'")
        else:
            open_palm_count = 0

            # ── Bounding box ──
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            padding = 20
            x1 = max(0, int(min(x_coords) * w) - padding)
            y1 = max(0, int(min(y_coords) * h) - padding)
            x2 = min(w, int(max(x_coords) * w) + padding)
            y2 = min(h, int(max(y_coords) * h) + padding)

            # Square crop
            bw, bh = x2 - x1, y2 - y1
            side = max(bw, bh)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

            roi = frame[y1:y2, x1:x2]

            # Prediction
            if roi.size > 0 and frame_count % PREDICT_EVERY == 0:
                reshaped, filtered = preprocess(roi)
                tensor = tf.constant(reshaped, dtype=tf.float32)
                prediction = model(tensor, training=False).numpy()
                label = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100

                if label == stable_label:
                    stable_count += 1
                else:
                    stable_label = label
                    stable_count = 1

                if (stable_count >= STABLE_THRESHOLD and
                    confidence >= MIN_CONFIDENCE and
                    now - last_capture_time >= CAPTURE_COOLDOWN):
                    if label != 0:
                        string += labels_dict[label]
                    last_capture_time = now
                    stable_count = 0
                    print(f"Captured: {labels_dict[label]} | Word: '{string}'")

            # ── Draw box ──
            color = (0, 0, 255) if confidence >= MIN_CONFIDENCE else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Confidence bar
            bar_width = int((x2 - x1) * confidence / 100)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 15), color, -1)

            # Prediction text
            cv2.putText(frame, f"{labels_dict[label]} {confidence:.0f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Stability bar
            cv2.putText(frame, f"Stable: {min(stable_count, STABLE_THRESHOLD)}/{STABLE_THRESHOLD}",
                        (x1, y2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if filtered is not None:
                cv2.imshow("Filtered Hand", filtered)

    # ── Open palm progress indicator ──
    if open_palm_count > 0:
        cv2.putText(frame, f"SPACE: {open_palm_count}/{PALM_HOLD}",
                    (w // 2 - 80, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ── Two hands indicator ──
    if num_hands == 2:
        cv2.putText(frame, "DELETE LAST...",
                    (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ── HUD ───────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Word: {string}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cooldown_left = max(0, CAPTURE_COOLDOWN - (now - last_capture_time))
    cv2.putText(frame, f"Next: {cooldown_left:.1f}s", (w - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if not hand_found and num_hands == 0:
        stable_count = 0
        open_palm_count = 0
        cv2.putText(frame, "Show your hand", (w // 2 - 110, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ── Guide ─────────────────────────────────────────────
    cv2.putText(frame, "Open palm=SPACE | 2 hands=DELETE | q=quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nFinal: {string}")
cap.release()
cv2.destroyAllWindows()