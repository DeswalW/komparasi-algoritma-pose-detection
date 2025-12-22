# PoseNet-like PyTorch demo (ResNet18 regressor)

Demo script that uses a ResNet18 backbone (pretrained) as a simple regressor to predict 2D keypoint coordinates (x,y) for pose detection. The outputs are normalized to [0,1] with a sigmoid and then scaled back to the original image size.

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
- This repository provides a minimal PoseNet-like regressor for inference/demo only. It does not include a training loop.
- The model resizes input images to 224x224 for the ResNet backbone. Predicted normalized coordinates are scaled back to the original image width and height.
- To train properly, you'll need a dataset with keypoint annotations and a training script that minimizes e.g. L2 loss on coordinates.

If you want, I can:
- Add a training script and a small synthetic dataset example.
- Add a checkpoint example or instructions to train on COCO/other datasets.
