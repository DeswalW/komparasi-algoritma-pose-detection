import cv2
import mediapipe as mp
import json

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load GT
with open('person_keypoints_default.json') as f:
    data = json.load(f)

# Get first annotation
first_ann = data['annotations'][0]
print(f"First annotation - image_id: {first_ann['image_id']}, num_keypoints: {first_ann['num_keypoints']}")

# Read video
cap = cv2.VideoCapture('video uji/SP_T_Duduk Berdiri_1.mp4')

frame_count = 0
detected_count = 0

while cap.isOpened() and frame_count < 10:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        detected_count += 1
        print(f"Frame {frame_count}: Hand detected with {len(results.multi_hand_landmarks[0].landmark)} landmarks")
    else:
        print(f"Frame {frame_count}: No hands detected")

cap.release()
print(f"\nDetected hands in {detected_count}/{frame_count} frames")
