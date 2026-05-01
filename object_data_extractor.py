import cv2
from ultralytics import YOLO
import mediapipe as mp
import csv
import os
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

#
video_path = "GroceryVideos/video-771_singular_display.MOV"
base_name = os.path.basename(video_path) 
video_name, _ = os.path.splitext(base_name) 
csv_filename = f"{video_name}_interactions.csv" 

model = YOLO("yolov8s-world.pt")
# More specific classes
model.set_classes(["bag of chips", "snack", "grocery item"])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Open CSV and video
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Hand_Side', 'Object_Name', 'Confidence'])

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_num += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_results = hands.process(img_rgb)
        yolo_results = model.track(frame, persist=True, conf=0.1, iou=0.5, verbose=False)
        
        # YOLO detects objects
        if len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes.xyxy.cpu()
            class_ids = yolo_results[0].boxes.cls.int().cpu().tolist()
            confidences = yolo_results[0].boxes.conf.cpu().tolist()
            
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                object_name = model.names[class_id]
                
                # Draws blue boxes around detected objects
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(frame, object_name, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Mediapipe detects hands
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if mp_results.multi_hand_landmarks and len(yolo_results[0].boxes) > 0:
            
            for hand_landmarks, handedness in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                # checking the Thumb (4), Index (8), and Middle (12) fingers
                fingertip_ids = [4, 8, 12]
                
                for box, class_id, conf in zip(boxes, class_ids, confidences):
                    x1, y1, x2, y2 = box 
                    object_name = model.names[class_id]
                    is_touching = False
                    
                    for tip_id in fingertip_ids:
                        finger = hand_landmarks.landmark[tip_id]
                        finger_x = int(finger.x * video_width)
                        finger_y = int(finger.y * video_height)
                        
                        # tiny green dots on those 3 fingers so you can see the math working
                        cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), cv2.FILLED)
                        
                        # If ANY of the three fingers enter the blue box...
                        if (x1 <= finger_x <= x2) and (y1 <= finger_y <= y2):
                            is_touching = True
                            break 
                    
                    if is_touching:
                        writer.writerow([frame_num, hand_label, object_name, round(conf, 2)])
                        
                        # Turn the blue box thick RED and save it
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                        cv2.putText(frame, f"TOUCHING: {object_name}", (int(x1), int(y1)-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Master Interaction Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()