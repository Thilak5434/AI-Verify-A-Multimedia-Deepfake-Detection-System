# 🎯 QUICK REFERENCE - FIND ANYTHING IN AI VERIFY

## Image Detection

| What | Search For | File | Line Look |
|------|-----------|------|-----------|
| Image model class | `ResNetBackbone` | `core.py` | Class definition |
| Image inference | `predict_image_bytes` | `core.py` | Function start |
| Image API endpoint | `@app.post("/analyze")` | `app.py` | Route handler |
| Image preprocessing | `eval_tf` | `core.py` | Transform pipeline |
| Image augmentation | `train_tf` | `core.py` | Augmentation pipeline |

---

## VIDEO Detection

| What | Search For | File |
|------|-----------|------|
| Frame-based detection | `predict_video_file` | `core.py` |
| Temporal detection (CNN-LSTM) | `CNNLSTM` | `core.py` |
| Load video model | `load_video_model` | `core.py` |
| Sequence prediction | `predict_video_with_cnnlstm` | `core.py` |
| Score fusion | `_fuse_video_scores` | `app.py` |

---

## AUDIO Detection

| What | Search For | File |
|------|-----------|------|
| Audio model | `AudioClassifier` | `core.py` |
| Audio inference | `predict_audio_file` | `core.py` |
| Mel-spectrogram | `audio_to_mel_spectrogram` | `core.py` |
| Load audio model | `load_audio_model` | `core.py` |
| Audio dataset | `AudioDataset` | `core.py` |

---

## Uncertainty & Calibration

| What | Search For | File |
|------|-----------|------|
| MC Dropout uncertainty | `mc_dropout_predict` | `core.py` |
| Temperature scaling | `TemperatureScaler` | `core.py` |
| Deterministic prediction | `deterministic_predict` | `core.py` |

---

## API & Results

| What | Search For | File |
|------|-----------|------|
| Main detection endpoint | `/analyze` | `app.py` |
| Health check endpoint | `/health` | `app.py` |
| Video analysis routing | `_analyze_video` | `app.py` |
| Audio analysis routing | `_analyze_audio` | `app.py` |
| Result caching | `RESULT_CACHE` | `app.py` |

---

## Libraries & Dependencies

```
CORE ML:
  PyTorch ................. Neural networks
  TorchVision ............ ResNet, transforms
  NumPy .................. Math operations
  scikit-learn .......... Metrics (F1, AUC, etc)

COMPUTER VISION:
  OpenCV (cv2) ......... Video frame extraction
  Pillow (PIL) ......... Image loading & augmentation

AUDIO:
  librosa .............. Mel-spectrogram extraction

API:
  FastAPI .............. REST framework
  
FRONTEND:
  HTML5, CSS3, JavaScript
  IndexedDB ............ Local storage
  ThreeJS .............. Custom cursor
```

---

## Common Tasks

### Find all detection Models
Search: `class.*Backbone\|class.*LSTM\|class.*Classifier`

### Find all Inference Functions  
Search: `def predict_`

### Find all Data Loading
Search: `class.*Dataset\|BinaryImageFolder`

### Find all Augmentation
Search: `class.*AdvancedJPEG\|RealisticBlur\|RealisticNoise`

### Find all Metrics
Search: `def compute_metrics\|ece_score\|roc_auc`

### Find Training Code
Open: `train.py`

---

## Model Files Location

All in `dataset/` folder:
- `media_detector.pt` → Image model
- `video_cnn_lstm.pt` → Video model  
- `audio_classifier.pt` → Audio model
- `media_detector_int8.pt` → Quantized image model
- `*.json` files → Metrics, threshold, temperature

---

## Frontend Pages

| Page | Purpose | Media Support |
|------|---------|---------------|
| `home.html` | Upload & scan interface | Image, Video, Audio |
| `scan.html` | Results display | All types |
| `about.html` | Project info | N/A |

---

## Command-Line Training

```bash
# Image training
python train.py train-image --data-dir <path> --epochs 10

# Video training
python train.py train-video --ffpp-dir <path> --epochs 5

# Audio training
python train.py train-audio --audio-path <path> --epochs 10
```

---

## Key Concepts

### IMAGE DETECTION
Model: ResNet50 + MC Dropout
Input: 224×224 RGB image
Output: Real (0) vs Fake (1) probability
Uncertainty: ✅ Yes (MC Dropout)

### VIDEO DETECTION
Models: ResNet50 (frames) + CNN-LSTM (temporal)
Approach: Dual pipeline with score fusion
Output: Real vs Fake with temporal stats
Uncertainty: ✅ Yes (MC Dropout on frames)

### AUDIO DETECTION
Model: AudioClassifier CNN
Input: 128×156 Mel-spectrogram
Output: Real vs Fake probability
Uncertainty: Single pass (no MC Dropout)

---

## Performance Tips

1. **GPU**: Models automatically use GPU if available
2. **Caching**: Results cached for 120 seconds
3. **Batch Processing**: API processes one file at a time
4. **Video**: Frame sampling reduces computation
5. **Quantization**: INT8 model available for deployment

---

## Debugging Commands

```python
# Check model loading
grep -n "Loaded model" app.py core.py

# Check inference functions
grep -n "def predict_" core.py

# Check metrics computation  
grep -n "compute_metrics\|f1_score" core.py

# Check augmentation
grep -n "class.*Advanced\|class.*Realistic" core.py

# Check API routing
grep -n "@app.\|def.*analyze" app.py
```

---

## File Size & Complexity

| File | Size | Complexity |
|------|------|-----------|
| `app.py` | ~250 lines | High (API + inference) |
| `core.py` | ~900 lines | Very High (Models + utils) |
| `train.py` | ~400 lines | High (Training) |
| `home.html` | ~100 lines | Medium |
| `scan.html` | ~150 lines | Medium |

---

**For more details, open**: `TECH_STACK_INDEX.md`

