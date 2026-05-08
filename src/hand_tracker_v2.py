# =============================================================================
# hand_tracker_v2.py
#
# PURPOSE:
#   Real-time hand + object interaction tracker for robotics training data.
#   Combines two AI models to detect when human hands touch objects in video:
#     - MediaPipe  → finds and tracks hands (21 keypoints per hand)
#     - YOLOv8     → detects and tracks objects (bounding boxes)
#
# OUTPUT:
#   1. Live annotated video window
#   2. CSV file logging every contact event with timestamp + flags
#   3. Annotated .mp4 video saved to disk automatically
#
# USAGE:
#   Edit the CONFIG section below, then run:
#     python src/hand_tracker_v2.py
#
# AUTHOR: Natalie Santos — Perspective Intelligence Internship
# =============================================================================

import cv2                                      # video reading, drawing, saving
from ultralytics import YOLO                    # object detection model
import mediapipe as mp                          # hand keypoint detection
import csv                                      # writing interaction logs
import os                                       # file path handling
import ssl                                      # fixes Mac SSL certificate errors
import time                                     # FPS calculation and timestamps
from collections import deque, defaultdict      # rolling window buffers

# Fix SSL certificate verification error on Mac
ssl._create_default_https_context = ssl._create_unverified_context


# =============================================================================
# CONFIG — edit these values without touching anything else
# =============================================================================

video_path       = "GroceryVideos/video-771_singular_display.MOV"

# Smoothing: how many frames to look back when stabilizing hand labels.
# Higher = more stable labels but slightly slower to respond to real changes.
SMOOTHING_WINDOW = 10

# Contact confirmation: how many of the last N frames need to show contact
# before we call it real. Higher = less flickering, slower to trigger.
CONTACT_FRAMES   = 6

# YOLO confidence threshold. Raise this (e.g. 0.35) if you are getting
# false detections of objects that are not really there.
YOLO_CONF        = 0.25

# How many pixels to expand each YOLO bounding box outward on all sides.
# This helps fingertips that are right at the edge of an object still register
# as touching. 20px is a good balance between sensitivity and false positives.
EXPAND_BOX_PX    = 20

# Set to True to save the annotated video as an .mp4 file automatically.
SAVE_VIDEO       = True


# =============================================================================
# FILE SETUP — automatically names output files based on the input video
# =============================================================================

base_name     = os.path.basename(video_path)
video_name, _ = os.path.splitext(base_name)
csv_filename  = f"{video_name}_interactions.csv"   # interaction log
out_filename  = f"{video_name}_demo_output.mp4"    # annotated video


# =============================================================================
# MODEL SETUP
# =============================================================================

# YOLOv8-World: an open-vocabulary object detector.
# You give it class names in plain English and it detects them without
# needing custom training.
model = YOLO("yolov8s-world.pt")
model.set_classes(["bag of chips", "snack", "grocery item", "bottle", "can", "box"])

# MediaPipe Hands: detects up to 2 hands and returns 21 3D keypoints per hand.
# Raising min_detection_confidence reduces ghost detections of non-hands.
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Custom drawing styles — teal landmarks with cyan connections
landmark_style   = mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3)
connection_style = mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2)


# =============================================================================
# SMOOTHING BUFFERS
# These store recent frame history to stabilize noisy per-frame predictions.
# =============================================================================

# label_history[hand_index] = deque of recent raw label strings
label_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))

# touch_history[(hand_index, object_id)] = deque of recent True/False values
touch_history = defaultdict(lambda: deque(maxlen=CONTACT_FRAMES))


# =============================================================================
# SESSION STATISTICS
# =============================================================================

session_stats = {
    "total_contacts":  0,
    "bimanual_frames": 0,
    "occluded_frames": 0,
    "start_time":      time.time()
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def smooth_label(hand_index, raw_label):
    """
    Converts MediaPipe raw hand label into a stable, correctly oriented label.

    WHY THIS IS NEEDED:
    MediaPipe uses a mirror convention so what it calls Right is actually the
    left hand on screen. We flip this first, then apply a majority vote over
    the last SMOOTHING_WINDOW frames so one bad frame cannot flip the label.

    Args:
        hand_index: 0 for first detected hand, 1 for second
        raw_label:  Left or Right as returned by MediaPipe

    Returns:
        Stable corrected label string: Left or Right
    """
    flipped = "Left" if raw_label == "Right" else "Right"
    label_history[hand_index].append(flipped)
    history = label_history[hand_index]
    return max(set(history), key=history.count)


def is_contact_confirmed(hand_index, obj_id, touching_now):
    """
    Determines whether a fingertip-object contact should be treated as real.

    WHY THIS IS NEEDED:
    Checking contact frame-by-frame causes flickering. This function requires
    contact to appear in at least half of the recent CONTACT_FRAMES frames
    before treating it as confirmed.

    Args:
        hand_index:   which hand (0 or 1)
        obj_id:       YOLO tracking ID for the object
        touching_now: whether contact was detected in the current frame

    Returns:
        True if contact is confirmed stable, False otherwise
    """
    touch_history[(hand_index, obj_id)].append(touching_now)
    window = touch_history[(hand_index, obj_id)]
    return sum(window) >= max(1, len(window) // 2)


def expand_box(x1, y1, x2, y2, px, w, h):
    """
    Expands a bounding box outward by px pixels, clamped to frame boundaries.

    WHY THIS IS NEEDED:
    YOLO boxes are tight to the visible object edge. When a fingertip touches
    the edge of an object it often lands just outside the box. Expanding by
    20px catches these edge cases.

    Returns:
        Tuple of expanded (x1, y1, x2, y2)
    """
    return (
        max(0, x1 - px),
        max(0, y1 - px),
        min(w, x2 + px),
        min(h, y2 + px)
    )


def is_occluded(hand_landmarks, boxes, video_width, video_height):
    """
    Checks whether a hand is at least partially hidden behind an object.

    HOW IT WORKS:
    Checks if the wrist (landmark 0) has entered any YOLO bounding box.
    If the wrist is inside an object box the hand is likely going behind it.
    This matches the Is_Occluded flag in the training dataset Excel sheet.

    Returns:
        True if occluded, False otherwise
    """
    wrist = hand_landmarks.landmark[0]
    wx    = int(wrist.x * video_width)
    wy    = int(wrist.y * video_height)
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1 <= wx <= x2 and y1 <= wy <= y2:
            return True
    return False


def draw_hud(frame, frame_num, fps, num_hands, is_bimanual, any_occluded, stats):
    """
    Draws a heads-up display panel in the top-left corner of the frame.
    Shows frame number, FPS, time, hand count, bimanual/occlusion status,
    and total contact events logged so far.
    """
    elapsed    = int(time.time() - stats["start_time"])
    mins, secs = divmod(elapsed, 60)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 220), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        (f"FRAME  {frame_num:>6}",                           (200, 200, 200)),
        (f"FPS    {fps:>6.1f}",                              (200, 200, 200)),
        (f"TIME   {mins:02d}:{secs:02d}",                   (200, 200, 200)),
        (f"HANDS  {num_hands}",                              (0, 255, 200)),
        (f"BIMANUAL   {'YES' if is_bimanual else 'NO '}",
            (0, 200, 255) if is_bimanual else (120, 120, 120)),
        (f"OCCLUDED   {'YES' if any_occluded else 'NO '}",
            (0, 165, 255) if any_occluded else (120, 120, 120)),
        (f"CONTACTS   {stats['total_contacts']:>4}",         (0, 255, 100)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)


def draw_contact_box(frame, x1, y1, x2, y2, label, object_name, conf):
    """
    Replaces the thin blue YOLO box with a thick red contact box + label.
    Draws a slightly larger outer box first for a glow effect, then the
    main red box, then a filled red banner showing the contact details.
    """
    cv2.rectangle(frame, (int(x1) - 3, int(y1) - 3),
                  (int(x2) + 3, int(y2) + 3), (0, 0, 180), 2)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                  (0, 0, 255), 3)
    banner_text = f"{label} TOUCHING: {object_name} ({conf:.0%})"
    (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame,
                  (int(x1), int(y1) - th - 12),
                  (int(x1) + tw + 8, int(y1)),
                  (0, 0, 255), -1)
    cv2.putText(frame, banner_text, (int(x1) + 4, int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)

    # CSV columns match the manual Excel annotation sheet format
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

    out_writer = None
    if SAVE_VIDEO:
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            out_filename, fourcc, fps_video, (video_width, video_height)
        )

    frame_num   = 0
    fps_display = 0.0
    prev_time   = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_num   += 1
        timestamp_s  = round(frame_num / fps_video, 2)

        now         = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time   = now

        # MediaPipe requires RGB input
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── STEP 1: Detect objects with YOLO ─────────────────────────────────
        # persist=True keeps consistent tracking IDs across frames
        yolo_results = model.track(
            frame, persist=True, conf=YOLO_CONF, iou=0.5, verbose=False
        )

        boxes, class_ids, confidences, track_ids = [], [], [], []

        if yolo_results and len(yolo_results[0].boxes) > 0:
            raw_boxes   = yolo_results[0].boxes.xyxy.cpu().tolist()
            class_ids   = yolo_results[0].boxes.cls.int().cpu().tolist()
            confidences = yolo_results[0].boxes.conf.cpu().tolist()
            track_ids   = (
                yolo_results[0].boxes.id.int().cpu().tolist()
                if yolo_results[0].boxes.id is not None
                else list(range(len(raw_boxes)))
            )

            # Expand boxes for better contact detection at edges
            boxes = [
                expand_box(*b, EXPAND_BOX_PX, video_width, video_height)
                for b in raw_boxes
            ]

            # Draw thin blue detection boxes
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                object_name = model.names[class_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 180, 0), 1)
                cv2.putText(frame, f"{object_name} {conf:.0%}",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

        # ── STEP 2: Detect hands with MediaPipe ──────────────────────────────
        mp_results = hands.process(img_rgb)

        num_hands    = len(mp_results.multi_hand_landmarks) if mp_results.multi_hand_landmarks else 0
        is_bimanual  = (num_hands == 2)
        any_occluded = False

        if mp_results.multi_hand_landmarks:
            if is_bimanual:
                session_stats["bimanual_frames"] += 1

            for hand_index, (hand_landmarks, handedness) in enumerate(
                zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness)
            ):
                raw_label    = handedness.classification[0].label
                stable_label = smooth_label(hand_index, raw_label)

                # Draw teal hand skeleton
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_style, connection_style
                )

                # Show stable Left/Right label above wrist
                wrist = hand_landmarks.landmark[0]
                wx    = int(wrist.x * video_width)
                wy    = int(wrist.y * video_height)
                cv2.putText(frame, stable_label, (wx, wy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 255), 2, cv2.LINE_AA)

                # Check for occlusion
                occluded = is_occluded(
                    hand_landmarks, boxes, video_width, video_height
                )
                if occluded:
                    any_occluded = True
                    session_stats["occluded_frames"] += 1
                    cv2.putText(frame, "OCCLUDED", (wx, wy - 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 165, 255), 2, cv2.LINE_AA)

                # Check all 5 fingertips against every detected object box
                fingertip_ids = [4, 8, 12, 16, 20]

                for box, class_id, conf, obj_id in zip(
                    boxes, class_ids, confidences, track_ids
                ):
                    x1, y1, x2, y2 = box
                    object_name    = model.names[class_id]
                    touching_now   = False

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
                        draw_contact_box(
                            frame, x1, y1, x2, y2,
                            stable_label, object_name, conf
                        )

        # ── STEP 3: HUD + display ─────────────────────────────────────────────
        draw_hud(frame, frame_num, fps_display,
                 num_hands, is_bimanual, any_occluded, session_stats)

        if out_writer:
            out_writer.write(frame)

        cv2.imshow("Hand Interaction Tracker v2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    print(f"\nDone!")
    print(f"  CSV saved   -> {csv_filename}")
    if SAVE_VIDEO:
        print(f"  Video saved -> {out_filename}")
    print(f"  Total contacts:  {session_stats['total_contacts']}")
    print(f"  Bimanual frames: {session_stats['bimanual_frames']}")
    print(f"  Occluded frames: {session_stats['occluded_frames']}")
