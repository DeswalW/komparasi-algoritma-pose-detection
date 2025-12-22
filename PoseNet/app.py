import argparse
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class PoseRegressor(torch.nn.Module):
	"""Simple PoseNet-like regressor built on ResNet18.

	The network outputs normalized (x, y) coordinates in [0,1] for each keypoint.
	Multiply by original image width/height to get pixel coordinates.
	"""

	def __init__(self, num_keypoints: int = 17, pretrained: bool = True):
		super().__init__()
		self.num_keypoints = num_keypoints
		# Prefer the new `weights` API when available to avoid deprecation warnings.
		try:
			weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
			self.backbone = models.resnet18(weights=weights)
		except Exception:
			# Fallback for older torchvision versions that expect `pretrained=`
			self.backbone = models.resnet18(pretrained=pretrained)
		in_f = self.backbone.fc.in_features
		# replace final fc with a regressor for 2*num_keypoints outputs
		self.backbone.fc = torch.nn.Linear(in_f, num_keypoints * 2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# output shape: (batch, num_keypoints*2)
		out = self.backbone(x)
		# normalize outputs to [0,1]
		out = torch.sigmoid(out)
		out = out.view(-1, self.num_keypoints, 2)
		return out

def load_model(device: torch.device, num_keypoints: int = 17, checkpoint: str = None) -> PoseRegressor:
	model = PoseRegressor(num_keypoints=num_keypoints, pretrained=True)
	if checkpoint:
		sd = torch.load(checkpoint, map_location=device)
		# try to load state dict gracefully
		if "state_dict" in sd:
			sd = sd["state_dict"]
		model.load_state_dict(sd)
	model = model.to(device)
	model.eval()
	return model

def preprocess_image(img_path: str, input_size: int = 224) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
	pil = Image.open(img_path).convert("RGB")
	orig_w, orig_h = pil.size
	transform = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	tensor = transform(pil)
	return tensor.unsqueeze(0), pil, (orig_w, orig_h)


def preprocess_frame(frame: np.ndarray, input_size: int = 224) -> Tuple[torch.Tensor, Tuple[int, int]]:
	"""Preprocess an OpenCV BGR frame for the model.

	Returns tensor (1,C,H,W) and original (w,h)
	"""
	# frame is BGR (H,W,3)
	h, w = frame.shape[:2]
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	pil = Image.fromarray(rgb)
	transform = transforms.Compose([
		transforms.Resize((input_size, input_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	tensor = transform(pil)
	return tensor.unsqueeze(0), (w, h)

def postprocess_preds(preds: np.ndarray, orig_size: Tuple[int, int], input_size: int = 224) -> np.ndarray:
	"""Convert normalized preds (in [0,1]) to original image pixel coords.

	preds: (num_keypoints, 2) in normalized coordinates relative to the model input.
	orig_size: (width, height)
	"""
	orig_w, orig_h = orig_size
	# since we resized to square input_size x input_size, scale by original dims
	xs = preds[:, 0] * orig_w
	ys = preds[:, 1] * orig_h
	keypoints = np.stack([xs, ys], axis=1)
	return keypoints

def draw_keypoints(img: np.ndarray, keypoints: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
	out = img.copy()
	for (x, y) in keypoints.astype(int):
		cv2.circle(out, (int(x), int(y)), radius=4, color=color, thickness=-1)
	return out

def run_on_image(model: torch.nn.Module, device: torch.device, img_path: str, output_path: str = None) -> str:
	input_size = 224
	tensor, pil_img, orig_size = preprocess_image(img_path, input_size=input_size)
	tensor = tensor.to(device)
	with torch.no_grad():
		preds = model(tensor)  # (1, num_kp, 2)
	preds = preds.cpu().numpy()[0]
	keypoints = postprocess_preds(preds, orig_size, input_size=input_size)

	# Read original image with OpenCV for drawing
	img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
	out_img = draw_keypoints(img_cv, keypoints)

	if output_path is None:
		base, ext = os.path.splitext(img_path)
		output_path = base + "_keypoints" + ext
	cv2.imwrite(output_path, out_img)
	return output_path

def run_on_webcam(model: torch.nn.Module, device: torch.device, camera_index: int = 0, display: bool = True, save_path: str = None, fps: int = 20):
	"""Run inference on webcam frames.

	- display: whether to show a window
	- save_path: if provided, record annotated video to this path
	"""
	cap = cv2.VideoCapture(int(camera_index))
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open camera index {camera_index}")

	# get camera size
	ret, frame = cap.read()
	if not ret:
		cap.release()
		raise RuntimeError("Failed to read frame from webcam")
	h, w = frame.shape[:2]

	writer = None
	if save_path:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(save_path, fourcc, float(fps), (w, h))

	print("Starting webcam. Press 'q' to quit.")
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			tensor, orig_size = preprocess_frame(frame, input_size=224)
			tensor = tensor.to(device)
			with torch.no_grad():
				preds = model(tensor)
			preds = preds.cpu().numpy()[0]
			keypoints = postprocess_preds(preds, orig_size, input_size=224)

			out_frame = draw_keypoints(frame, keypoints)

			if writer:
				writer.write(out_frame)

			if display:
				cv2.imshow("PoseNet Webcam", out_frame)
				# waitKey 1 for real-time; 1ms
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	finally:
		cap.release()
		if writer:
			writer.release()
		if display:
			cv2.destroyAllWindows()

def parse_args():
	p = argparse.ArgumentParser(description="Simple PoseNet-like pose regressor demo (PyTorch).")
	p.add_argument("image", nargs="?", help="Path to input image. If omitted and --webcam is set, webcam will be used.")
	p.add_argument("--checkpoint", help="Optional model checkpoint (state_dict)")
	p.add_argument("--out", help="Optional path to save output image with keypoints (or output video when using --webcam)")
	p.add_argument("--num-keypoints", type=int, default=17, help="Number of keypoints to predict")
	p.add_argument("--webcam", action="store_true", help="Enable webcam mode instead of single image")
	p.add_argument("--camera-index", type=int, default=0, help="Webcam camera index (default 0)")
	p.add_argument("--no-display", dest="display", action="store_false", help="Do not display window (useful for headless recording)")
	p.add_argument("--fps", type=int, default=20, help="FPS for output video when saving webcam recording")
	return p.parse_args()

def main():
	args = parse_args()
	# If no image provided and webcam not explicitly disabled, default to webcam
	if not args.image and not args.webcam:
		print("No image provided; defaulting to webcam mode. Use --no-display for headless recording or --out to save video.")
		args.webcam = True
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	model = load_model(device, num_keypoints=args.num_keypoints, checkpoint=args.checkpoint)
	if args.webcam:
		# run webcam mode
		run_on_webcam(model, device, camera_index=args.camera_index, display=args.display, save_path=args.out, fps=args.fps)
		if args.out:
			print(f"Saved recorded webcam video to: {args.out}")
	else:
		if not args.image:
			print("No image provided and webcam mode not set. Exiting.")
			return
		out_path = run_on_image(model, device, args.image, output_path=args.out)
		print(f"Saved output with keypoints to: {out_path}")

if __name__ == "__main__":
	main()