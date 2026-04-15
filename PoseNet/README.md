# PoseNet-like PyTorch demo

This folder now supports two runtime modes:

1. `keypointrcnn_resnet50_fpn` (default): COCO-pretrained keypoint detector from torchvision.
2. `resnet18_regressor` (legacy): simple regressor demo that requires your own checkpoint for good results.

If you want strong out-of-the-box accuracy, use the default pretrained keypoint R-CNN mode.

Bahasa / Indonesian quick usage:

1. Install dependencies (preferably in a venv):

```powershell
python -m pip install -r requirements.txt
```

2. Run the script on an image (GPU will be used automatically if available):

```powershell
python app.py path\to\image.jpg --out path\to\out.jpg
```

Optionally provide a checkpoint (PyTorch state_dict) trained for keypoints:

```powershell
python app.py path\to\image.jpg --checkpoint path\to\checkpoint.pth --out out.jpg
```

Choose model mode explicitly:

```powershell
# Recommended: pretrained pose model
python app.py path\to\image.jpg --model-type keypointrcnn_resnet50_fpn --out out.jpg

# Legacy demo regressor (requires checkpoint for usable quality)
python app.py path\to\image.jpg --model-type resnet18_regressor --checkpoint path\to\checkpoint.pth --out out.jpg
```

Webcam usage
------------

You can run inference directly from your webcam. Example that displays a window and does not save video:

```powershell
python app.py --webcam
```

To record the annotated webcam output to a video file (MP4):

```powershell
python app.py --webcam --out out_video.mp4
```

If you're running headless (no display) and want to record only, disable the display:

```powershell
python app.py --webcam --no-display --out out_video.mp4
```

Notes:
- `keypointrcnn_resnet50_fpn` is pretrained on COCO and does not require your own checkpoint for inference.
- `resnet18_regressor` is a minimal baseline and does not include a training loop in this folder.
- For `resnet18_regressor`, you need a dataset with keypoint annotations and a training script that minimizes e.g. L2 loss on coordinates.

If you want, I can:
- Add a training script and a small synthetic dataset example.
- Add a checkpoint example or instructions to train on COCO/other datasets.
