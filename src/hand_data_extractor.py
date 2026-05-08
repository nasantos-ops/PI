import cv2
import mediapipe as mp
import csv
import os

# the video we are tracking
video_path = "GroceryVideos/video-771_singular_display.MOV"

# auto-name the csv file so it matches the video
base_name = os.path.basename(video_path) 
video_name, _ = os.path.splitext(base_name) 
csv_filename = f"{video_name}_coordinates.csv" 

# set up mediapipe for 2 hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# open the csv to save our math
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # top row of the spreadsheet
    writer.writerow(['Frame', 'Hand_Num', 'Left_or_Right', 'Joint_ID', 'X', 'Y', 'Z'])

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        success, frame = cap.read()
        if not success:
            break # video is done
        
        frame_num += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # if it sees hands and knows left vs right
        if results.multi_hand_landmarks and results.multi_handedness:
            
            # loop through the hands
            for idx, (landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                
                hand_label = handedness.classification[0].label 
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                # grab the xyz for all 21 joints and save to csv
                for joint_id, joint in enumerate(landmarks.landmark):
                    writer.writerow([
                        frame_num, idx, hand_label, joint_id, joint.x, joint.y, joint.z
                    ])

        cv2.imshow("Tracking", frame)
        
        # hit q to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()