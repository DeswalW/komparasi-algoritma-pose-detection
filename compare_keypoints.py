import json
import csv
import numpy as np
from pathlib import Path

def load_json_file(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_bbox_diagonal(keypoints_full):
    """Calculate bounding box diagonal from keypoints"""
    if not keypoints_full or len(keypoints_full) == 0:
        return 1.0
    
    # Extract x, y coordinates (skip visibility)
    coords = np.array([keypoints_full[i:i+3][:2] for i in range(0, len(keypoints_full), 3)])
    if len(coords) == 0:
        return 1.0
    
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    return max(diagonal, 1.0)

def compare_keypoints():
    """Compare ground truth keypoints with detected keypoints and save to CSV"""
    
    # File paths
    gt_file = "person_keypoints_default.json"
    results_file = "results_SP_T_Duduk Berdiri_1.json"
    output_csv = "keypoint_comparison.csv"
    
    # Validate files exist
    if not Path(gt_file).exists():
        print(f"❌ File not found: {gt_file}")
        return
    if not Path(results_file).exists():
        print(f"❌ File not found: {results_file}")
        return
    
    # Load data
    print("📂 Loading data...")
    gt_data = load_json_file(gt_file)
    results_data = load_json_file(results_file)
    
    # Extract annotations
    gt_annotations = gt_data.get('annotations', [])
    pred_annotations = results_data.get('annotations', [])
    
    # Create mapping for ground truth by image_id
    gt_by_image = {}
    for ann in gt_annotations:
        image_id = ann.get('image_id', ann.get('id'))
        gt_by_image[image_id] = ann
    
    # Prepare CSV data
    csv_rows = []
    
    print("🔄 Comparing keypoints...")
    
    # Keypoint indices mapping (12 keypoints from MediaPipe Pose)
    keypoint_names = [
        "R_Shoulder",    # 0 (MediaPipe index 5)
        "L_Shoulder",    # 1 (MediaPipe index 6)
        "R_Elbow",       # 2 (MediaPipe index 7)
        "L_Elbow",       # 3 (MediaPipe index 8)
        "R_Wrist",       # 4 (MediaPipe index 9)
        "L_Wrist",       # 5 (MediaPipe index 10)
        "R_Hip",         # 6 (MediaPipe index 11)
        "L_Hip",         # 7 (MediaPipe index 12)
        "R_Knee",        # 8 (MediaPipe index 13)
        "L_Knee",        # 9 (MediaPipe index 14)
        "R_Ankle",       # 10 (MediaPipe index 15)
        "L_Ankle"        # 11 (MediaPipe index 16)
    ]
    
    # Process each predicted annotation
    for pred_ann in pred_annotations:
        image_id = pred_ann.get('image_id', 0)
        pred_keypoints = pred_ann.get('keypoints', [])
        pck_score = pred_ann.get('pck', 0)
        oks_score = pred_ann.get('oks', 0)
        
        # Get corresponding ground truth
        gt_ann = gt_by_image.get(image_id, {})
        gt_keypoints_full = gt_ann.get('keypoints', [])
        gt_keypoints = gt_keypoints_full[:36] if len(gt_keypoints_full) >= 36 else gt_keypoints_full
        
        # Calculate bbox diagonal for PCK threshold
        bbox_diagonal = calculate_bbox_diagonal(gt_keypoints_full)
        pck_threshold = 0.2 * bbox_diagonal
        
        # Process each keypoint
        for kp_idx in range(len(keypoint_names)):
            kp_name = keypoint_names[kp_idx]
            
            # Extract predicted keypoint (x, y pairs)
            pred_x = pred_keypoints[kp_idx * 2] if kp_idx * 2 < len(pred_keypoints) else None
            pred_y = pred_keypoints[kp_idx * 2 + 1] if kp_idx * 2 + 1 < len(pred_keypoints) else None
            
            # Extract ground truth keypoint (x, y, visibility)
            gt_x = gt_keypoints[kp_idx * 3] if kp_idx * 3 < len(gt_keypoints) else None
            gt_y = gt_keypoints[kp_idx * 3 + 1] if kp_idx * 3 + 1 < len(gt_keypoints) else None
            gt_vis = gt_keypoints[kp_idx * 3 + 2] if kp_idx * 3 + 2 < len(gt_keypoints) else 0
            
            # Calculate error distance and PCK if both exist
            error_distance = None
            is_correct_pck = None
            
            if (pred_x is not None and pred_y is not None and 
                gt_x is not None and gt_y is not None and gt_vis > 0):
                
                error_distance = calculate_distance(pred_x, pred_y, gt_x, gt_y)
                is_correct_pck = 1 if error_distance <= pck_threshold else 0
            
            # Create CSV row
            row = {
                'Frame_ID': image_id,
                'Keypoint_Index': kp_idx,
                'Keypoint_Name': kp_name,
                'GT_X': round(gt_x, 2) if gt_x is not None else None,
                'GT_Y': round(gt_y, 2) if gt_y is not None else None,
                'GT_Visibility': int(gt_vis) if gt_vis is not None else None,
                'Pred_X': round(pred_x, 2) if pred_x is not None else None,
                'Pred_Y': round(pred_y, 2) if pred_y is not None else None,
                'Error_Distance': round(error_distance, 2) if error_distance is not None else None,
                'PCK_Threshold': round(pck_threshold, 2),
                'Is_Correct_PCK': is_correct_pck,
                'Frame_PCK_Score': round(pck_score, 4),
                'Frame_OKS_Score': round(oks_score, 4)
            }
            csv_rows.append(row)
    
    # Write to CSV
    print(f"💾 Writing to CSV: {output_csv}")
    
    if csv_rows:
        fieldnames = csv_rows[0].keys()
        
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"✅ CSV saved successfully!")
        print(f"\n📊 Summary:")
        print(f"   - Total rows: {len(csv_rows)}")
        print(f"   - Total frames: {len(pred_annotations)}")
        print(f"   - Keypoints per frame: {len(keypoint_names)}")
        print(f"   - Output file: {Path(output_csv).absolute()}")
        
        # Calculate statistics
        error_distances = [row['Error_Distance'] for row in csv_rows if row['Error_Distance'] is not None]
        correct_pck_count = sum(1 for row in csv_rows if row['Is_Correct_PCK'] == 1)
        total_valid = sum(1 for row in csv_rows if row['Is_Correct_PCK'] is not None)
        
        if error_distances:
            print(f"\n📈 Statistics:")
            print(f"   - Mean error distance: {np.mean(error_distances):.2f} px")
            print(f"   - Std error distance: {np.std(error_distances):.2f} px")
            print(f"   - Min error distance: {np.min(error_distances):.2f} px")
            print(f"   - Max error distance: {np.max(error_distances):.2f} px")
            
            if total_valid > 0:
                pck_accuracy = (correct_pck_count / total_valid) * 100
                print(f"   - PCK Accuracy: {pck_accuracy:.2f}% ({correct_pck_count}/{total_valid})")
        
        print(f"\n📋 CSV Columns:")
        print(f"   Frame_ID - Frame number (1-{len(pred_annotations)})")
        print(f"   Keypoint_Index - Keypoint number (0-11)")
        print(f"   Keypoint_Name - Name of the keypoint")
        print(f"   GT_X, GT_Y - Ground truth coordinates")
        print(f"   GT_Visibility - Ground truth visibility flag")
        print(f"   Pred_X, Pred_Y - Predicted coordinates")
        print(f"   Error_Distance - Euclidean distance between GT and prediction")
        print(f"   PCK_Threshold - 20% of bounding box diagonal")
        print(f"   Is_Correct_PCK - 1 if error <= threshold, 0 otherwise")
        print(f"   Frame_PCK_Score - Overall PCK score for the frame")
        print(f"   Frame_OKS_Score - Overall OKS score for the frame")
        
    else:
        print("❌ No data to write!")

if __name__ == "__main__":
    compare_keypoints()
