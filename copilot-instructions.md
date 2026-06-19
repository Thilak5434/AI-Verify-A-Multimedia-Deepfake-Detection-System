# Copilot Instructions for AI Generated Media Detector

This repository implements a deepfake detection system that classifies images, videos, and audio as real or AI-generated.

## Architecture Overview

### Core Components

**Backend** (`backend/`):
- **FastAPI server** (`app.py`): REST API with `/analyze` endpoint for media classification. Auto-reloads models on file changes; implements 2-minute result caching.
- **Inference logic** (`core.py`): Contains model architectures and prediction pipelines:
  - **VisionBackbone**: MobileNetV3-Small or ViT-B16 classifier (224×224 images)
  - **CNNLSTM**: CNN+LSTM model for video frame sequences (12-20 frames)
  - **Prediction**: Uses MC Dropout (20 passes) for uncertainty estimation
  - **Calibration**: Temperature scaling and threshold optimization on validation split
- **Training** (`train.py`): CLI-based training with subcommands:
  - `train-image`: Train image classifier with 80/20 train/val split
  - `train-video`: Train CNNLSTM on FaceForensics++ videos
  - `cross-dataset`: Evaluate generalization across datasets
  - `quantize`: INT8 quantization for deployment

**Frontend** (`home.html`, `home.css`):
- Single-page app with file upload, drag-and-drop, and IndexedDB caching
- Displays results: label, confidence, fake probability, uncertainty

**Dataset** (`dataset/`):
- Directory structure: `real/` and `fake/` subdirectories with image files
- Models stored here: `media_detector.pt`, `video_cnn_lstm.pt`
- Metadata: `threshold.json` (decision threshold), `temperature.json` (calibration), `metrics.json` (validation metrics)

## Build, Test, and Lint

### Run Backend Server
```powershell
cd backend
pip install -r requirements.txt
python app.py
```
Server runs on `http://localhost:3000`. Frontend calls `POST /analyze` with media files.

### Train Image Detector
```powershell
cd backend
python train.py train-image --dataset "..\dataset" --arch mobilenet --epochs 8 --adv-train --out "..\dataset\media_detector.pt" --temp-out "..\dataset\temperature.json" --metrics-out "..\dataset\metrics.json"
```
- Architecture: `mobilenet` (default) or `vit`
- Batch size: 16 (default)
- Learning rate: 1e-4 (default)
- `--adv-train` enables FGSM adversarial training for robustness

### Train Video Model (CNNLSTM)
```powershell
cd backend
python train.py train-video --ffpp "..\ffpp_data" --epochs 4 --frames 8 --video-out "..\dataset\video_cnn_lstm.pt"
```
- Expects FaceForensics++ structure: `ffpp_data/original_sequences/*.mp4` and `ffpp_data/manipulated_sequences/*.mp4`
- Frames per video: 8 (default)
- Max videos to use: 60 (default, 0 = all)

### Cross-Dataset Evaluation
```powershell
cd backend
python train.py cross-dataset --train-on "..\dataset" --test-on "..\dataset" --adv-train --cross-out "..\dataset\cross_dataset_metrics.json"
```
Train on one dataset (e.g., FaceForensics+), test on another (e.g., Celeb-DF) to measure generalization.

### INT8 Quantization
```powershell
cd backend
python train.py quantize --in-model "..\dataset\media_detector.pt" --out-model "..\dataset\media_detector_int8.pt"
```
Produces a quantized model for efficient inference.

## Key Conventions

### Dataset Structure
- Images must be in `dataset/real/` and `dataset/fake/` directories
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Videos must follow FaceForensics++ layout: `ffpp_data/original_sequences/` and `ffpp_data/manipulated_sequences/`

### Model Training
- Always use `seed_all(42)` for reproducibility
- 80/20 train/val split with fixed random seed
- Image size: 224×224 (ImageNet standard)
- Metrics tracked: accuracy, precision, recall, F1, AUC, confusion matrix, ECE (Expected Calibration Error)
- Temperature scaling adjusts confidence calibration; optimal temperature learned via cross-entropy
- Decision threshold optimized to maximize F1 on validation set

### Inference Pipeline
1. **Preprocessing**: Resize to 224×224, normalize with ImageNet stats ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
2. **Prediction**: Run 20 MC Dropout passes to get mean probability and standard deviation
3. **Calibration**: Apply temperature scaling and decision threshold
4. **Video Fusion**: For videos, blend frame-level predictions (MobileNet) with sequence-level predictions (CNNLSTM) at 85/15 ratio
5. **Output**: label (Real/AI Generated), confidence, fake_probability, uncertainty_std

### API Response Format
```json
{
  "result": "Real",
  "confidence": 0.95,
  "fake_probability": 0.05,
  "uncertainty_std": 0.02,
  "hash_sha256": "...",
  "cached": false,
  "details": {
    "label_id": 0,
    "label": "Real",
    "temporal": { "frame_probs": [...] },
    "fusion": { "frame_model_p_fake": 0.05, "video_model_p_fake": 0.03, ... }
  }
}
```

### Threshold and Calibration Files
- `threshold.json`: Decision boundary to maximize F1; typically around 0.5-0.6
- `temperature.json`: Scaling factor (T>1 means overconfident predictions)
- Both files are auto-generated during training; manually load from model directory during inference

### Error Handling
- Empty uploads return 400 with "empty upload"
- Unsupported media types return 415
- Processing errors return 500 with exception message
- Invalid model paths raise RuntimeError on server startup

### Adversarial Training
- FGSM perturbations with epsilon=2/255
- Gradient-based attack applied during training to improve robustness
- Use `--adv-train` flag to enable (increases training time ~10-20%)

### Explainability
- Grad-CAM heatmaps available for MobileNet (not ViT) via `gradcam_image()`
- Returns base64-encoded JPEG; used to visualize decision regions
- Not used in current frontend but available for debugging model behavior
