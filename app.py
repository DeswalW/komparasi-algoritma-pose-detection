import cv2
import mediapipe as mp
import json
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Initialize MediaPipe Pose (not Hands!)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

# Load ground truth annotations
def load_ground_truth(json_path):
    """Load COCO format annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Extract keypoints by image_id
def get_gt_keypoints_by_image(coco_data):
    """Create a dictionary mapping image_id to keypoints"""
    gt_keypoints = {}
    
    # Handle both direct list and COCO format
    if isinstance(coco_data, dict):
        annotations = coco_data.get('annotations', [])
    else:
        annotations = coco_data
    
    for annotation in annotations:
        image_id = annotation['image_id']
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)  # [x, y, visibility]
        gt_keypoints[image_id] = keypoints
    return gt_keypoints

# Calculate PCK (Percentage of Correct Keypoints)
def calculate_pck(predicted_kpts, gt_kpts, gt_kpts_full=None, threshold=0.2):
    """
    Calculate PCK metric
    predicted_kpts: [N, 2] array of predicted points
    gt_kpts: [N, 2] array of ground truth points
    gt_kpts_full: [N, 3] array with visibility info
    threshold: normalized by bounding box diagonal (default 0.2 * diagonal)
    """
    if len(predicted_kpts) == 0 or len(gt_kpts) == 0:
        return 0.0
    
    # Calculate bounding box diagonal
    gt_bbox = np.array([gt_kpts[:, 0].min(), gt_kpts[:, 1].min(),
                        gt_kpts[:, 0].max(), gt_kpts[:, 1].max()])
    bbox_width = gt_bbox[2] - gt_bbox[0]
    bbox_height = gt_bbox[3] - gt_bbox[1]
    bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
    
    if bbox_diagonal == 0:
        return 0.0
    
    threshold_dist = threshold * bbox_diagonal
    
    correct_keypoints = 0
    total_visible = 0
    
    for i in range(min(len(predicted_kpts), len(gt_kpts))):
        # Check visibility from full keypoints if available
        is_visible = True
        if gt_kpts_full is not None and i < len(gt_kpts_full):
            is_visible = gt_kpts_full[i, 2] > 0
        
        if is_visible:
            total_visible += 1
            dist = np.linalg.norm(predicted_kpts[i, :2] - gt_kpts[i, :2])
            if dist <= threshold_dist:
                correct_keypoints += 1
    
    if total_visible == 0:
        return 0.0
    
    return correct_keypoints / total_visible

# Calculate OKS (Object Keypoint Similarity)
def calculate_oks(predicted_kpts, gt_kpts, gt_kpts_full=None, area=1.0):
    """
    Calculate OKS metric
    predicted_kpts: [N, 2] array of predicted points
    gt_kpts: [N, 2] array of ground truth points
    gt_kpts_full: [N, 3] array with visibility info
    area: bounding box area
    """
    if len(predicted_kpts) == 0 or len(gt_kpts) == 0:
        return 0.0
    
    # Sigma values for pose keypoints (COCO standard - arm and leg keypoints)
    sigma = np.array([0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.68])
    
    num_keypoints = min(len(predicted_kpts), len(gt_kpts), len(sigma))
    
    distances = []
    visibilities = []
    
    for i in range(num_keypoints):
        # Check visibility
        is_visible = True
        if gt_kpts_full is not None and i < len(gt_kpts_full):
            is_visible = gt_kpts_full[i, 2] > 0
        
        if is_visible:
            dist = np.linalg.norm(predicted_kpts[i, :2] - gt_kpts[i, :2])
            distances.append(dist)
            visibilities.append(1)
        else:
            distances.append(0)
            visibilities.append(0)
    
    if sum(visibilities) == 0:
        return 0.0
    
    oks_values = []
    for i in range(num_keypoints):
        if visibilities[i] > 0:
            s = sigma[i] if i < len(sigma) else 0.1
            oks = np.exp(-(distances[i]**2) / (2 * (s**2) * area))
            oks_values.append(oks)
    
    if len(oks_values) == 0:
        return 0.0
    
    return np.mean(oks_values)

# Process video and calculate metrics
def process_video(video_path, gt_keypoints_dict, output_dir):
    """Process video and calculate OKS and PCK metrics"""
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    video_filename = Path(video_path).stem
    output_video_path = os.path.join(output_dir, f"comparison_{video_filename}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 1  # Start from 1, not 0
    pck_scores = []
    oks_scores = []
    predicted_annotations = []  # Store predictions for JSON
    frame_errors = []  # Store per-keypoint errors
    
    print(f"\nProcessing video: {video_path}")
    print(f"Video Properties - FPS: {fps}, Resolution: {width}x{height}, Total Frames: {total_frames}")
    print("=" * 60)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Get ground truth for this frame
        if frame_idx not in gt_keypoints_dict:
            frame_idx += 1
            continue
        
        gt_kpts = gt_keypoints_dict[frame_idx]
        
        # Convert image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Extract MediaPipe Pose keypoints
        if results.pose_landmarks:
            # Convert MediaPipe landmarks to pixel coordinates
            h, w, c = image.shape
            predicted_kpts = []
            for landmark in results.pose_landmarks.landmark:
                predicted_kpts.append([landmark.x * w, landmark.y * h, landmark.z])
            predicted_kpts = np.array(predicted_kpts)
            
            # Filter to only arm and leg keypoints (indices for COCO pose)
            # Arms: 5,6,7,8,9,10 (shoulders, elbows, wrists)
            # Legs: 11,12,13,14,15,16 (hips, knees, ankles)
            # Let's map to ground truth (13 keypoints)
            arm_leg_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 12 keypoints
            
            if len(predicted_kpts) > 16:
                predicted_kpts_filtered = predicted_kpts[arm_leg_indices]
            else:
                predicted_kpts_filtered = predicted_kpts
            
            # Calculate metrics
            pck = calculate_pck(predicted_kpts_filtered[:, :2], gt_kpts[:, :2], gt_kpts)
            
            # For OKS, we need bounding box area
            bbox = np.array([gt_kpts[:, 0].min(), gt_kpts[:, 1].min(),
                           gt_kpts[:, 0].max(), gt_kpts[:, 1].max()])
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area = max(area, 1.0)  # Avoid division by zero
            
            oks = calculate_oks(predicted_kpts_filtered[:, :2], gt_kpts[:, :2], gt_kpts, area)
            
            pck_scores.append(pck)
            oks_scores.append(oks)
            
            # Store predicted keypoints as COCO annotation format
            predicted_keypoints_flat = predicted_kpts_filtered[:, :2].flatten().tolist()
            num_kpts = len(predicted_kpts_filtered)
            
            annotation = {
                'id': len(predicted_annotations),
                'image_id': frame_idx,
                'category_id': 1,
                'keypoints': predicted_keypoints_flat,
                'num_keypoints': num_kpts,
                'pck': pck,
                'oks': oks
            }
            predicted_annotations.append(annotation)
            
            # Calculate per-keypoint errors
            for i in range(min(len(predicted_kpts_filtered), len(gt_kpts))):
                error_dist = np.linalg.norm(predicted_kpts_filtered[i, :2] - gt_kpts[i, :2])
                frame_errors.append({
                    'frame_id': frame_idx,
                    'keypoint_idx': i,
                    'error': float(error_dist)
                })
            
            # Draw predictions and ground truth
            for i in range(min(len(predicted_kpts_filtered), len(gt_kpts))):
                pred_pt = predicted_kpts_filtered[i]
                gt_pt = gt_kpts[i]
                
                # Predicted keypoint (red)
                x, y = int(pred_pt[0]), int(pred_pt[1])
                cv2.circle(image_bgr, (x, y), 3, (0, 0, 255), -1)
                
                # Ground truth keypoint (green)
                gt_x, gt_y = int(gt_pt[0]), int(gt_pt[1])
                cv2.circle(image_bgr, (gt_x, gt_y), 3, (0, 255, 0), -1)
                
                # Draw line between predicted and ground truth
                cv2.line(image_bgr, (x, y), (gt_x, gt_y), (255, 0, 0), 1)
            
            # Draw pose connections
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display metrics
            text_y = 30
            cv2.putText(image_bgr, f"PCK: {pck:.4f}", (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_bgr, f"OKS: {oks:.4f}", (10, text_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_bgr, f"FPS: {fps:.1f}", (10, text_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No pose detected - just show ground truth
            h, w, c = image.shape
            for i, gt_pt in enumerate(gt_kpts):
                gt_x, gt_y = int(gt_pt[0]), int(gt_pt[1])
                cv2.circle(image_bgr, (gt_x, gt_y), 3, (0, 255, 0), -1)
            
            text_y = 30
            cv2.putText(image_bgr, "No pose detected", (10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image_bgr, f"FPS: {fps:.1f}", (10, text_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(image_bgr, f"Frame: {frame_idx}", (10, image_bgr.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame to output video
        out.write(image_bgr)
        
        cv2.imshow('MediaPipe Hand Pose Evaluation', image_bgr)
        
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Release video writer
    out.release()
    print(f"Video comparison saved to: {output_video_path}")
    
    # Calculate average metrics
    if len(pck_scores) > 0:
        avg_pck = np.mean(pck_scores)
        std_pck = np.std(pck_scores)
    else:
        avg_pck = std_pck = 0.0
    
    if len(oks_scores) > 0:
        avg_oks = np.mean(oks_scores)
        std_oks = np.std(oks_scores)
    else:
        avg_oks = std_oks = 0.0
    
    return {
        'pck_scores': pck_scores,
        'oks_scores': oks_scores,
        'avg_pck': avg_pck,
        'std_pck': std_pck,
        'avg_oks': avg_oks,
        'std_oks': std_oks,
        'num_frames': len(pck_scores),
        'fps': fps,
        'predicted_annotations': predicted_annotations,
        'frame_errors': frame_errors,
        'output_video': output_video_path
    }

# Save results to JSON
def save_results_to_json(results, output_path):
    """Save detected keypoints and metrics to JSON"""
    output_data = {
        'info': {
            'description': 'MediaPipe Pose Detection Results',
            'version': '1.0'
        },
        'annotations': results['predicted_annotations'],
        'metrics': {
            'PCK': {
                'average': results['avg_pck'],
                'std_dev': results['std_pck'],
                'all_scores': results['pck_scores']
            },
            'OKS': {
                'average': results['avg_oks'],
                'std_dev': results['std_oks'],
                'all_scores': results['oks_scores']
            },
            'total_frames': results['num_frames']
        },
        'keypoint_errors': results['frame_errors']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_path}")

# Create visualization plots
def create_evaluation_plots(results, output_dir):
    """Create comprehensive evaluation plots"""
    
    pck_scores = results['pck_scores']
    oks_scores = results['oks_scores']
    frame_errors = results['frame_errors']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. PCK and OKS over frames
    ax1 = fig.add_subplot(gs[0, :])
    frames = np.arange(1, len(pck_scores) + 1)
    ax1.plot(frames, pck_scores, label='PCK', linewidth=2, color='#1f77b4', alpha=0.8)
    ax1.plot(frames, oks_scores, label='OKS', linewidth=2, color='#ff7f0e', alpha=0.8)
    ax1.axhline(y=results['avg_pck'], color='#1f77b4', linestyle='--', alpha=0.5, label=f'Avg PCK: {results["avg_pck"]:.4f}')
    ax1.axhline(y=results['avg_oks'], color='#ff7f0e', linestyle='--', alpha=0.5, label=f'Avg OKS: {results["avg_oks"]:.4f}')
    ax1.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('PCK and OKS Scores per Frame', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. PCK Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(pck_scores, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.axvline(results['avg_pck'], color='red', linestyle='--', linewidth=2, label=f'Mean: {results["avg_pck"]:.4f}')
    ax2.axvline(np.median(pck_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(pck_scores):.4f}')
    ax2.set_xlabel('PCK Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('PCK Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. OKS Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(oks_scores, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax3.axvline(results['avg_oks'], color='red', linestyle='--', linewidth=2, label=f'Mean: {results["avg_oks"]:.4f}')
    ax3.axvline(np.median(oks_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(oks_scores):.4f}')
    ax3.set_xlabel('OKS Score', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('OKS Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Keypoint Error Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    errors = [e['error'] for e in frame_errors]
    ax4.hist(errors, bins=40, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}px')
    ax4.axvline(np.median(errors), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}px')
    ax4.set_xlabel('Error Distance (pixels)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Keypoint Error Distance Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Error by Keypoint Index
    ax5 = fig.add_subplot(gs[2, 1])
    keypoint_indices = [e['keypoint_idx'] for e in frame_errors]
    errors_by_kpt = [[] for _ in range(max(keypoint_indices) + 1)]
    for e in frame_errors:
        errors_by_kpt[e['keypoint_idx']].append(e['error'])
    
    mean_errors = [np.mean(errs) if errs else 0 for errs in errors_by_kpt]
    ax5.bar(range(len(mean_errors)), mean_errors, color='#d62728', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Keypoint Index', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Mean Error (pixels)', fontsize=11, fontweight='bold')
    ax5.set_title('Mean Error Distance per Keypoint', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MediaPipe Pose Detection - Evaluation Metrics', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = os.path.join(output_dir, 'evaluation_plots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")
    
    # Also save individual stats plot
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Summary statistics
    stats_text = f"""
    EVALUATION SUMMARY
    
    PCK (Percentage of Correct Keypoints):
      • Average: {results['avg_pck']:.4f}
      • Std Dev: {results['std_pck']:.4f}
      • Min: {np.min(pck_scores):.4f}
      • Max: {np.max(pck_scores):.4f}
    
    OKS (Object Keypoint Similarity):
      • Average: {results['avg_oks']:.4f}
      • Std Dev: {results['std_oks']:.4f}
      • Min: {np.min(oks_scores):.4f}
      • Max: {np.max(oks_scores):.4f}
    
    Error Distance:
      • Mean: {np.mean(errors):.2f} pixels
      • Std Dev: {np.std(errors):.2f} pixels
      • Min: {np.min(errors):.2f} pixels
      • Max: {np.max(errors):.2f} pixels
    
    Frames Processed: {results['num_frames']}
    """
    
    axes[0].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].axis('off')
    
    # Comparison bar chart
    metrics = ['PCK', 'OKS']
    means = [results['avg_pck'], results['avg_oks']]
    stds = [results['std_pck'], results['std_oks']]
    
    x = np.arange(len(metrics))
    axes[1].bar(x, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e'], 
                alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Average Metrics with Std Dev', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, fontsize=11, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[1].text(i, mean + std + 0.02, f'{mean:.4f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    plt.suptitle('Evaluation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Summary saved to: {output_path2}")
    
    plt.close('all')

# Main execution
if __name__ == "__main__":
    # Path to ground truth and video
    script_dir = Path(__file__).parent
    gt_path = script_dir / "person_keypoints_default.json"
    video_dir = script_dir / "video uji"
    
    # Load ground truth
    print("Loading ground truth annotations...")
    gt_data = load_ground_truth(gt_path)
    gt_keypoints = get_gt_keypoints_by_image(gt_data)
    print(f"Loaded {len(gt_keypoints)} frames of ground truth")
    
    # Process each video in the directory
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        
        if video_files:
            for video_file in video_files:
                results = process_video(str(video_file), gt_keypoints, script_dir)
                
                # Print results
                print(f"\n{'='*60}")
                print(f"Results for: {video_file.name}")
                print(f"{'='*60}")
                print(f"Number of frames processed: {results['num_frames']}")
                print(f"Video FPS: {results['fps']:.1f}")
                print(f"\nPCK (Percentage of Correct Keypoints):")
                print(f"  Average: {results['avg_pck']:.4f}")
                print(f"  Std Dev: {results['std_pck']:.4f}")
                print(f"\nOKS (Object Keypoint Similarity):")
                print(f"  Average: {results['avg_oks']:.4f}")
                print(f"  Std Dev: {results['std_oks']:.4f}")
                print(f"{'='*60}\n")
                
                # Save results to JSON
                output_json = script_dir / f"results_{video_file.stem}.json"
                save_results_to_json(results, output_json)
                
                # Create evaluation plots
                create_evaluation_plots(results, script_dir)
        else:
            print(f"No .mp4 files found in {video_dir}")
    else:
        print(f"Video directory not found: {video_dir}")