import cv2
from ultralytics import YOLO
import mediapipe as mp
import csv
import os
import ssl
from collections import deque, defaultdict

ssl._create_default_https_context = ssl._create_unverified_context

# ── CONFIG ────────────────────────────────────────────────────────────────────
video_path = "GroceryVideos/video-771_singular_display.MOV"
SMOOTHING_WINDOW = 7   # frames to vote over for hand label stability
CONTACT_FRAMES   = 4   # how many of the last N frames need contact to confirm it

# ── SETUP ─────────────────────────────────────────────────────────────────────
base_name   = os.path.basename(video_path)
video_name, _ = os.path.splitext(base_name)
csv_filename = f"{video_name}_interactions.csv"

model = YOLO("yolov8s-world.pt")
model.set_classes(["bag of chips", "snack", "grocery item"])

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=2)
mp_draw  = mp.solutions.drawing_utils

# ── SMOOTHING STATE ───────────────────────────────────────────────────────────
# For each hand slot (0 or 1), keep a rolling window of raw labels
label_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))

# For each (hand_slot, object_id) pair, keep a rolling window of touch booleans
touch_history = defaultdict(lambda: deque(maxlen=CONTACT_FRAMES))


def smooth_label(hand_index, raw_label):
    """
    MediaPipe mirrors hands — 'Right' in its world = left side of screen.
    We flip it, then stabilise with a majority vote over recent frames.
    """
    # Flip because MediaPipe uses a mirror convention
    flipped = "Left" if raw_label == "Right" else "Right"
    label_history[hand_index].append(flipped)

    # Majority vote — whichever label appears most in the window wins
    history = label_history[hand_index]
    return max(set(history), key=history.count)


def is_contact_confirmed(hand_index, obj_id, touching_now):
    """
    Only call a contact 'real' if it appears in at least half of the recent
    window — kills single-frame flickers.
    """
    touch_history[(hand_index, obj_id)].append(touching_now)
    window = touch_history[(hand_index, obj_id)]
    return sum(window) >= max(1, len(window) // 2)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Hand_Side', 'Object_Name', 'Confidence'])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        exit()

    frame_num    = 0
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_num += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_results  = hands.process(img_rgb)
        yolo_results = model.track(frame, persist=True, conf=0.1, iou=0.5, verbose=False)

        # ── Parse YOLO detections (guard against empty results) ───────────────
        boxes, class_ids, confidences, track_ids = [], [], [], []
        if yolo_results and len(yolo_results[0].boxes) > 0:
            boxes       = yolo_results[0].boxes.xyxy.cpu().tolist()
            class_ids   = yolo_results[0].boxes.cls.int().cpu().tolist()
            confidences = yolo_results[0].boxes.conf.cpu().tolist()

            # Use tracking ID if available, otherwise fall back to box index
            if yolo_results[0].boxes.id is not None:
                track_ids = yolo_results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = list(range(len(boxes)))

            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                object_name = model.names[class_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(frame, object_name, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # ── MediaPipe hand detection ──────────────────────────────────────────
        if mp_results.multi_hand_landmarks:
            for hand_index, (hand_landmarks, handedness) in enumerate(
                zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness)
            ):
                raw_label    = handedness.classification[0].label
                stable_label = smooth_label(hand_index, raw_label)  # ← smoothed + flipped

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Label above the wrist (landmark 0)
                wrist = hand_landmarks.landmark[0]
                wx = int(wrist.x * video_width)
                wy = int(wrist.y * video_height)
                cv2.putText(frame, stable_label, (wx, wy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # ── Contact detection ─────────────────────────────────────────
                fingertip_ids = [4, 8, 12]  # Thumb, Index, Middle

                for (box, class_id, conf, obj_id) in zip(boxes, class_ids, confidences, track_ids):
                    x1, y1, x2, y2 = box
                    object_name = model.names[class_id]

                    # Check raw touch this frame
                    touching_now = False
                    for tip_id in fingertip_ids:
                        finger   = hand_landmarks.landmark[tip_id]
                        finger_x = int(finger.x * video_width)
                        finger_y = int(finger.y * video_height)

                        cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), cv2.FILLED)

                        if (x1 <= finger_x <= x2) and (y1 <= finger_y <= y2):
                            touching_now = True
                            break

                    # Only confirm contact if it's been stable across recent frames
                    confirmed = is_contact_confirmed(hand_index, obj_id, touching_now)

                    if confirmed:
                        writer.writerow([frame_num, stable_label, object_name, round(conf, 2)])
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        cv2.putText(frame, f"{stable_label} TOUCHING: {object_name}",
                                    (int(x1), int(y1) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Master Interaction Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! Saved to {csv_filename}")