import json
import cv2

# Load GT data
with open('person_keypoints_default.json') as f:
    data = json.load(f)

# Check video
cap = cv2.VideoCapture('video uji/SP_T_Duduk Berdiri_1.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Get image IDs from GT
image_ids = sorted(set(a['image_id'] for a in data['annotations']))

print(f'Video total frames: {frame_count}')
print(f'GT image_ids range: {min(image_ids)} to {max(image_ids)}')
print(f'Total unique image_ids: {len(image_ids)}')
print(f'First 10 image_ids: {image_ids[:10]}')
print(f'Last 10 image_ids: {image_ids[-10:]}')
