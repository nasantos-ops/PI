import cv2
from ultralytics import YOLO
import mediapipe as mp
import csv
import os
import ssl
import time
from collections import deque, defaultdict

ssl._create_default_https_context = ssl._create_unverified_context

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — tweak these without touching anything else
# ══════════════════════════════════════════════════════════════════════════════
video_path       = "GroceryVideos/video-771_singular_display.MOV"
SMOOTHING_WINDOW = 10   # frames for hand-label majority vote (higher = more stable)
CONTACT_FRAMES   = 6    # frames needed to confirm a touch (higher = less flicker)
YOLO_CONF        = 0.25 # raise this if you're getting false object detections
EXPAND_BOX_PX    = 20   # expand YOLO boxes by this many pixels (helps contact detection)
SAVE_VIDEO       = True # set True to export a demo .mp4 alongside the CSV

# ══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════════════════
base_name    = os.path.basename(video_path)
video_name, _ = os.path.splitext(base_name)
csv_filename = f"{video_name}_interactions.csv"
out_filename = f"{video_name}_demo_output.mp4"

model = YOLO("yolov8s-world.pt")
model.set_classes(["bag of chips", "snack", "grocery item", "bottle", "can", "box"])

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,   # raised — reduces ghost hand detections
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils
landmark_style  = mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3)
connection_style = mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2)

# ── Smoothing buffers ──────────────────────────────────────────────────────
label_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
touch_history = defaultdict(lambda: deque(maxlen=CONTACT_FRAMES))

# ── Session stats (shown in HUD) ──────────────────────────────────────────
session_stats = {
    "total_contacts": 0,
    "bimanual_frames": 0,
    "occluded_frames": 0,
    "start_time": time.time()
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def smooth_label(hand_index, raw_label):
    """Flip MediaPipe's mirrored convention, then majority-vote over recent frames."""
    flipped = "Left" if raw_label == "Right" else "Right"
    label_history[hand_index].append(flipped)
    history = label_history[hand_index]
    return max(set(history), key=history.count)


def is_contact_confirmed(hand_index, obj_id, touching_now):
    """Return True only if touch has been detected in ≥ half of recent frames."""
    touch_history[(hand_index, obj_id)].append(touching_now)
    window = touch_history[(hand_index, obj_id)]
    return sum(window) >= max(1, len(window) // 2)


def expand_box(x1, y1, x2, y2, px, w, h):
    """Expand a bounding box by px pixels, clamped to frame bounds."""
    return (
        max(0, x1 - px),
        max(0, y1 - px),
        min(w, x2 + px),
        min(h, y2 + px)
    )


def is_occluded(hand_landmarks, boxes, video_width, video_height):
    """
    Simple occlusion heuristic: if the wrist (landmark 0) is inside a YOLO
    box, the hand is likely partially hidden behind the object.
    """
    wrist = hand_landmarks.landmark[0]
    wx = int(wrist.x * video_width)
    wy = int(wrist.y * video_height)
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 <= wx <= x2 and y1 <= wy <= y2:
            return True
    return False


def draw_hud(frame, frame_num, fps, num_hands, is_bimanual, any_occluded, stats):
    """Draw a clean heads-up display in the top-left corner."""
    h, w = frame.shape[:2]
    elapsed = int(time.time() - stats["start_time"])
    mins, secs = divmod(elapsed, 60)

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 220), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        (f"FRAME  {frame_num:>6}",          (200, 200, 200)),
        (f"FPS    {fps:>6.1f}",              (200, 200, 200)),
        (f"TIME   {mins:02d}:{secs:02d}",   (200, 200, 200)),
        (f"HANDS  {num_hands}",              (0, 255, 200)),
        (f"BIMANUAL   {'YES' if is_bimanual else 'NO '}",
                            (0, 200, 255) if is_bimanual else (120, 120, 120)),
        (f"OCCLUDED   {'YES' if any_occluded else 'NO '}",
                            (0, 165, 255) if any_occluded else (120, 120, 120)),
        (f"CONTACTS   {stats['total_contacts']:>4}", (0, 255, 100)),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)


def draw_contact_box(frame, x1, y1, x2, y2, label, object_name, conf):
    """Draw a pulsing-style thick red box with a clean label banner."""
    # Outer glow effect (draw slightly larger box first)
    cv2.rectangle(frame, (int(x1)-3, int(y1)-3), (int(x2)+3, int(y2)+3),
                  (0, 0, 180), 2)
    # Main red box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                  (0, 0, 255), 3)
    # Label banner above box
    banner_text = f"{label} TOUCHING: {object_name} ({conf:.0%})"
    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (int(x1), int(y1) - th - 12),
                  (int(x1) + tw + 8, int(y1)), (0, 0, 255), -1)
    cv2.putText(frame, banner_text, (int(x1) + 4, int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Frame', 'Timestamp_s', 'Hand_Side', 'Object_Name',
        'Confidence', 'Is_Bimanual', 'Is_Occluded'
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        exit()

    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional: write output video
    out_writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_filename, fourcc, fps_video,
                                     (video_width, video_height))

    frame_num  = 0
    fps_display = 0.0
    prev_time  = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_num += 1
        timestamp_s = round(frame_num / fps_video, 2)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FPS calc ──────────────────────────────────────────────────────
        now = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── YOLO object detection ─────────────────────────────────────────
        yolo_results = model.track(frame, persist=True,
                                   conf=YOLO_CONF, iou=0.5, verbose=False)
        boxes, class_ids, confidences, track_ids = [], [], [], []

        if yolo_results and len(yolo_results[0].boxes) > 0:
            raw_boxes   = yolo_results[0].boxes.xyxy.cpu().tolist()
            class_ids   = yolo_results[0].boxes.cls.int().cpu().tolist()
            confidences = yolo_results[0].boxes.conf.cpu().tolist()
            track_ids   = (yolo_results[0].boxes.id.int().cpu().tolist()
                           if yolo_results[0].boxes.id is not None
                           else list(range(len(raw_boxes))))

            # Expand boxes slightly — big win for contact reliability
            boxes = [expand_box(*b, EXPAND_BOX_PX, video_width, video_height)
                     for b in raw_boxes]

            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                object_name = model.names[class_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 180, 0), 1)
                cv2.putText(frame, f"{object_name} {conf:.0%}",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

        # ── MediaPipe hand detection ──────────────────────────────────────
        mp_results = hands.process(img_rgb)

        num_hands    = len(mp_results.multi_hand_landmarks) if mp_results.multi_hand_landmarks else 0
        is_bimanual  = num_hands == 2
        any_occluded = False

        if mp_results.multi_hand_landmarks:
            if is_bimanual:
                session_stats["bimanual_frames"] += 1

            for hand_index, (hand_landmarks, handedness) in enumerate(
                zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness)
            ):
                raw_label    = handedness.classification[0].label
                stable_label = smooth_label(hand_index, raw_label)

                # Draw skeleton
                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS,
                                       landmark_style, connection_style)

                # Hand label above wrist
                wrist = hand_landmarks.landmark[0]
                wx = int(wrist.x * video_width)
                wy = int(wrist.y * video_height)
                cv2.putText(frame, stable_label, (wx, wy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 255), 2, cv2.LINE_AA)

                # Occlusion check
                occluded = is_occluded(hand_landmarks, boxes, video_width, video_height)
                if occluded:
                    any_occluded = True
                    session_stats["occluded_frames"] += 1
                    cv2.putText(frame, "OCCLUDED", (wx, wy - 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 165, 255), 2, cv2.LINE_AA)

                # ── Contact detection per object ──────────────────────────
                fingertip_ids = [4, 8, 12, 16, 20]  # all 5 fingertips now

                for box, class_id, conf, obj_id in zip(
                        boxes, class_ids, confidences, track_ids):
                    x1, y1, x2, y2 = box
                    object_name = model.names[class_id]
                    touching_now = False

                    for tip_id in fingertip_ids:
                        finger   = hand_landmarks.landmark[tip_id]
                        finger_x = int(finger.x * video_width)
                        finger_y = int(finger.y * video_height)

                        cv2.circle(frame, (finger_x, finger_y),
                                   4, (0, 255, 150), cv2.FILLED)

                        if x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                            touching_now = True
                            break

                    confirmed = is_contact_confirmed(hand_index, obj_id, touching_now)

                    if confirmed:
                        session_stats["total_contacts"] += 1
                        writer.writerow([
                            frame_num, timestamp_s, stable_label,
                            object_name, round(conf, 2),
                            is_bimanual, occluded
                        ])
                        draw_contact_box(frame, x1, y1, x2, y2,
                                         stable_label, object_name, conf)

        # ── HUD overlay ───────────────────────────────────────────────────
        draw_hud(frame, frame_num, fps_display,
                 num_hands, is_bimanual, any_occluded, session_stats)

        # ── Output ────────────────────────────────────────────────────────
        if out_writer:
            out_writer.write(frame)

        cv2.imshow("Hand Interaction Tracker v2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Done!")
    print(f"   CSV saved  → {csv_filename}")
    if SAVE_VIDEO:
        print(f"   Video saved → {out_filename}")
    print(f"   Total contact events logged: {session_stats['total_contacts']}")
    print(f"   Bimanual frames: {session_stats['bimanual_frames']}")
    print(f"   Occluded frames: {session_stats['occluded_frames']}")
