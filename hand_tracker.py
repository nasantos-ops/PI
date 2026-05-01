import cv2
import mediapipe as mp

# Setting up the tracking part of the code using mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2) #Looking for 2 hands only in the video (This will help with our dataset)
mp_draw = mp.solutions.drawing_utils #hand landmarks

# Load video
cap = cv2.VideoCapture("GroceryVideos/video-779_singular_display 2.mov")

if not cap.isOpened():
    print("Could not open video.")
    exit()

while True: #
    success, frame = cap.read() #grabs each frame of the video and repeats
    if not success:
        print("Could not read frame. Ending video.")
        break
    #This part is needed because openCV uses BGR format while mediapipe uses RGB format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = hands.process(img_rgb) #This part will actually process the video
    #This loop = if hands are found, skeleton will appear
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", frame) #shows the actual video
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
