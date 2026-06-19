# AI Verify - Tech Stack & Module Index

**Last Updated**: April 2026  
**Purpose**: Searchable reference for tech stack, libraries, algorithms, and module locations

---

## 🔍 QUICK SEARCH GUIDE

### By Functionality
- **Want IMAGE DETECTION?** → Search: `ResNet50`, `ResNetBackbone`, `predict_image_bytes` in `backend/core.py`
- **Want VIDEO DETECTION?** → Search: `CNN-LSTM`, `CNNLSTM`, `predict_video_file`, `predict_video_with_cnnlstm` in `backend/core.py`
- **Want AUDIO DETECTION?** → Search: `AudioClassifier`, `audio_to_mel_spectrogram`, `predict_audio_file` in `backend/core.py`
- **Want API ENDPOINTS?** → Search: `/analyze`, `@app.post` in `backend/app.py`
- **Want TRAINING?** → Check `backend/train.py`

---

## 📚 COMPLETE TECH STACK

### Backend Framework
- **Framework**: FastAPI
- **Location**: `backend/app.py`
- **Purpose**: REST API server for media detection
- **Endpoints**: `/analyze`, `/health`

### Deep Learning Libraries
| Library | Version | Purpose | Location |
|---------|---------|---------|----------|
| PyTorch | Latest | Neural network training & inference | `backend/core.py`, `backend/train.py` |
| TorchVision | Latest | Pre-trained models (ResNet50, ResNet152) | `backend/core.py` |
| NumPy | Latest | Numerical operations | `backend/core.py` |
| scikit-learn | Latest | Model metrics (F1, AUC, precision, recall) | `backend/core.py` |

### Computer Vision
| Library | Purpose | Location |
|---------|---------|----------|
| OpenCV (cv2) | Video frame extraction | `backend/core.py` |
| Pillow (PIL) | Image loading & augmentation | `backend/core.py` |

### Audio Processing
| Library | Purpose | Location |
|---------|---------|----------|
| librosa | Mel-spectrogram extraction | `backend/core.py` (audio_to_mel_spectrogram) |

### Frontend
| Technology | Purpose | Location |
|------------|---------|----------|
| HTML5 | Structure | `home.html`, `scan.html`, `about.html` |
| CSS3 | Styling | `home.css` |
| JavaScript (ES6+) | Interactivity, API calls | `home.html`, `scan.html` |
| IndexedDB | Client-side storage | `scan.html` |
| ThreeJS (Tubes) | Custom cursor animation | `home.html`, `scan.html` |

---

## 🧠 DETECTION ALGORITHMS

### IMAGE DETECTION MODEL
```
ARCHITECTURE: ResNet50/ResNet152/ResNet101 Backbone + MC Dropout + Temperature Scaling
LOCATION: backend/core.py → ResNetBackbone class
PURPOSE: Classify images as Real (0) or AI-Generated/Fake (1)

INFERENCE PIPELINE:
  1. Decode image bytes → PIL Image
  2. Apply transforms (normalize, resize to 224x224)
  3. Run deterministic prediction (single pass)
  4. Apply Temperature Scaler for calibration
  5. Run MC Dropout (25 passes) for uncertainty estimation
  6. Compute confidence interval [p_fake - uncertainty, p_fake + uncertainty]

KEY FEATURES:
  - Pre-trained on ImageNet (ResNet50_Weights.IMAGENET1K_V1)
  - Dropout rate: 0.3 for uncertainty estimation
  - MC Dropout: 25 forward passes for probability distribution
  - Temperature scaling: Post-hoc calibration via TemperatureScaler class

INFERENCE FUNCTION: predict_image_bytes() in backend/core.py
```

### VIDEO DETECTION MODEL (Dual Pipeline)
```
ARCHITECTURE 1 - Frame-Level (ResNet50):
  - Same as IMAGE DETECTION
  - Applied to N sampled frames from video
  - Aggregates frame predictions: 60% mean + 25% p90 + 15% max

ARCHITECTURE 2 - Temporal (CNN-LSTM):
  - CNN: ResNet50 feature extractor (2048 features per frame)
  - LSTM: 2-layer bidirectional LSTM (hidden=256)
  - Head: Classification network with dropout
  
LOCATION: backend/core.py → CNNLSTM class
PURPOSE: Capture temporal inconsistencies in AI-generated videos

INFERENCE PIPELINE:
  1. Extract N frames uniformly from video (using OpenCV)
  2. Frame-level analysis: ResNet50 on each frame
  3. Temporal analysis: CNNLSTM on sequence of frames
  4. Score Fusion: Weighted combination (55% frame model + 45% video model)
  
INFERENCE FUNCTIONS:
  - predict_video_file() → Frame-level analysis
  - predict_video_with_cnnlstm() → Temporal analysis
  - _fuse_video_scores() → Score fusion logic
  - Location: backend/core.py & backend/app.py
```

### AUDIO DETECTION MODEL
```
ARCHITECTURE: AudioClassifier CNN on Mel-Spectrograms
LOCATION: backend/core.py → AudioClassifier class
PURPOSE: Detect AI voice cloning and synthetic audio

INPUT: Mel-spectrogram (1 channel, 128 mels, 156 time steps)
PROCESSING:
  1. Raw audio (WAV/MP3/FLAC) → Load with librosa at 16 kHz
  2. Compute Mel-spectrogram using librosa.feature.melspectrogram()
  3. Convert to dB scale: librosa.power_to_db()
  4. Normalize: Clip to [-80, 0] dB, scale to [0, 1]
  5. Pad/trim to (128, 156) for fixed input

ARCHITECTURE:
  - 4 Conv2D blocks (1→32→64→128→256 filters)
  - BatchNorm + ReLU after each conv
  - MaxPool2d (2x2) after each block
  - Global Average Pooling → Flatten to 256 features
  - Classification head: FC(256→128)→ReLU→Dropout→FC(128→2)

INFERENCE FUNCTION: predict_audio_file() in backend/core.py
```

---

## 🎯 data augmentation

```
IMAGE AUGMENTATION:
  - AdvancedJPEGCompression: Simulate JPEG compression artifacts (q: 20-95)
  - RealisticBlur: Motion blur + Gaussian blur
  - RealisticNoise: Gaussian noise + low-light simulation
  - ColorDistortion: Brightness, contrast, color shifts
  - RandomErasing: Random erasure of image regions

LOCATION: backend/core.py → train_tf() function

PURPOSE: Improve model robustness to real-world compression and artifacts
```

---

## 📊 EVALUATION METRICS

### Computed During Training/Evaluation
| Metric | Library | Purpose |
|--------|---------|---------|
| F1 Score | scikit-learn | Balance precision & recall |
| Accuracy | scikit-learn | Overall correctness |
| Precision | scikit-learn | False positive rate |
| Recall | scikit-learn | False negative rate (sensitivity) |
| AUC-ROC | scikit-learn | Model discrimination ability |
| Confusion Matrix | scikit-learn | TP/FP/TN/FN breakdown |
| ECE (Expected Calibration Error) | Custom | Confidence calibration |

**Location**: `compute_metrics()` and `ece_score()` in `backend/core.py`

---

## 🔐 CALIBRATION & UNCERTAINTY

### Temperature Scaling
```
ALGORITHM: Post-hoc calibration via temperature parameter
CLASS: TemperatureScaler (backend/core.py)
PURPOSE: Adjust model confidence to match accuracy
METHOD: Scale logits by temperature T before softmax

Formula: p_calibrated = softmax(logits / T)
Range: T ∈ [0.05, 10.0]
```

### Monte Carlo Dropout Uncertainty
```
ALGORITHM: MC Dropout (Gal & Ghahramani, 2016)
FUNCTION: mc_dropout_predict() in backend/core.py
PURPOSE: Estimate model uncertainty via stochastic inference
METHOD: Run N forward passes with dropout enabled, compute statistics

Output: Mean probability and standard deviation
Image: 25 passes
Video frames: 12 passes
```

---

## 🔗 API ENDPOINTS

### `/analyze` (POST)
```
ENDPOINT: POST /analyze
INPUT: Multipart file upload (image/video/audio)

MEDIA TYPE ROUTING:
  - image/* → predict_image_bytes()
  - video/* → _analyze_video() → predict_video_file() + predict_video_with_cnnlstm()
  - audio/* → _analyze_audio() → predict_audio_file()

CACHING:
  - Key: model_version + SHA256(file)
  - TTL: 120 seconds
  - Max size: 300 entries

OUTPUT: JSON with:
  - label_id: 0=Real, 1=Fake
  - label: Human-readable label
  - confidence: 0-1 confidence score
  - fake_probability: Probability of being AI-generated
  - uncertainty_std: Uncertainty estimate
  - uncertainty_interval: [min, max] confidence range
  - deepfake_type: Type of deepfake (GAN, lip-sync, voice clone, etc)
  - hash_sha256: File hash for deduplication
  - cached: Whether result was cached
```

### `/health` (GET)
```
ENDPOINT: GET /health
PURPOSE: Check server status
RETURNS: Model loaded status, version info
```

---

## 📁 FILE STRUCTURE & LOCATIONS

### Backend Files
```
backend/
  ├── app.py
  │   ├── REST API endpoints (/analyze, /health)
  │   ├── Model loading (ResNet50, CNN-LSTM, AudioClassifier)
  │   ├── Result caching logic
  │   ├── Media routing (image/video/audio)
  │   └── DETECTION ENTRY POINTS:
  │       - predict_image_bytes()
  │       - _analyze_video() → Frame + Temporal analysis
  │       - _analyze_audio()
  │
  ├── core.py
  │   ├── IMAGE MODELS: ResNetBackbone, TemperatureScaler
  │   ├── VIDEO MODELS: CNNLSTM, FFPPVideoDataset
  │   ├── AUDIO MODELS: AudioClassifier, AudioDataset
  │   ├── DATA LOADING: BinaryImageFolder, FFPPVideoDataset, AudioDataset
  │   ├── AUGMENTATION: AdvancedJPEGCompression, RealisticBlur, RealisticNoise, ColorDistortion
  │   ├── INFERENCE:
  │   │   - predict_image_bytes()
  │   │   - predict_video_file()
  │   │   - predict_video_with_cnnlstm()
  │   │   - predict_audio_file()
  │   │   - mc_dropout_predict()
  │   │   - deterministic_predict()
  │   ├── UTILITIES:
  │   │   - build_inference_pack()
  │   │   - sample_video_frames()
  │   │   - audio_to_mel_spectrogram()
  │   │   - compute_metrics()
  │   │   - ece_score()
  │   │   - provenance_hash()
  │   └── TRAINING: AdvancedTrainer (PGD adversarial training)
  │
  └── train.py
      ├── Command-line training interface
      ├── Image training (ResNet50/ResNet152)
      ├── Video training (CNN-LSTM)
      ├── Audio training (AudioClassifier)
      ├── Cross-dataset evaluation
      └── Model quantization (INT8)

Frontend Files
├── home.html → Upload & scan interface
├── scan.html → Results & analysis display
├── about.html → Project information
└── home.css → Styling

Dataset Files
├── media_detector.pt → Image detection model weights
├── video_cnn_lstm.pt → Video CNN-LSTM model weights
├── audio_classifier.pt → Audio classification model weights
├── media_detector_int8.pt → Quantized image model
├── metrics.json → Image model evaluation metrics
├── video_metrics.json → Video model metrics
├── audio_metrics.json → Audio model metrics
├── temperature.json → Temperature scaling value
├── threshold.json → Decision threshold
└── cross_dataset_metrics.json → Cross-dataset evaluation results
```

---

## 🚀 RUNNING THE SYSTEM

### Start Backend API
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 3000
```

### Train Models
```bash
# Image detection
python train.py train-image --data-dir <dataset_dir> --epochs 10

# Video detection
python train.py train-video --ffpp-dir <ffpp_dataset_dir> --epochs 5

# Audio detection
python train.py train-audio --audio-path <audio_dir> --epochs 10

# Cross-dataset evaluation
python train.py cross-eval --data-dir <dataset_dir> --model-dir dataset
```

### Start Frontend
Open `home.html` in browser or serve via HTTP

---

## 🔑 KEY SEARCH TERMS

Use these keywords to find specific functionality:

| Keyword | Finds | Location |
|---------|-------|----------|
| `ResNet50` | Image model architecture | `core.py` |
| `CNNLSTM` | Video temporal model | `core.py` |
| `AudioClassifier` | Audio detection model | `core.py` |
| `mc_dropout` | Uncertainty estimation | `core.py` |
| `temperature` | Calibration | `core.py` |
| `melspectrogram` | Audio feature extraction | `core.py` |
| `/analyze` | Main detection endpoint | `app.py` |
| `predict_image` | Image inference | `core.py` |
| `predict_video` | Video inference | `core.py` |
| `predict_audio` | Audio inference | `core.py` |
| `train_tf` | Augmentation pipeline | `core.py` |
| `eval_tf` | Inference preprocessing | `core.py` |
| `AdvancedJPEG` | Compression augmentation | `core.py` |
| `RESULT_CACHE` | Result caching mechanism | `app.py` |
| `@app.post` | API endpoints | `app.py` |

---

## 📝 NOTES FOR DEVELOPERS

1. **Model Loading**: All models are loaded at startup in `app.py`
2. **GPU/CPU**: Automatically selects GPU if available, falls back to CPU
3. **Model Files**: Stored in `dataset/` directory, automatically reloaded if updated
4. **Caching**: Results cached by (model_version + file_hash) for 120 seconds
5. **Uncertainty**: MC Dropout uncertainty available for all media types
6. **Calibration**: Temperature scaling applied to improve confidence reliability
7. **Adversarial Training**: PGD attack available for model robustness (in train.py)

---

## 📚 REFERENCES

- Gal & Ghahramani (2016): [Uncertainty in Deep Learning via Dropout](https://arxiv.org/abs/1506.02142)
- Guo et al. (2017): [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- Madry et al. (2018): [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
- FaceForensics++ Dataset: https://github.com/ondyari/FaceForensics
- ResNet Paper: He et al. (2015): Deep Residual Learning for Image Recognition

