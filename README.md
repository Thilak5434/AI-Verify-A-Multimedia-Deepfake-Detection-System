"# AI-Verify-A-Multimedia-Deepfake-Detection-System" 
🚀 Quick Start - Deepfake Detection v3.0.0
Installation
cd backend
pip install -r requirements.txt
Train Your Model (Recommended: ResNet50)
Step 1: Prepare Dataset
Organize your images:

dataset/
├── real/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── fake/
    ├── deepfake1.jpg
    ├── deepfake2.png
    └── ...
Minimum: 200 images total (100 real, 100 fake) Recommended: 1000+ images for good generalization

Step 2: Train Model
# Basic training (12 epochs, ResNet50)
python train.py train-image \
  --dataset "../dataset" \
  --arch resnet50 \
  --epochs 12 \
  --batch 16 \
  --adv-train

# This will create:
# - media_detector.pt (model weights)
# - temperature.json (calibration)
# - threshold.json (decision boundary)
# - metrics.json (evaluation results)
Training time: ~30-60 minutes on modern GPU (RTX3060+)

Step 3: Test on Your Data
# Evaluate on separate test set
python train.py eval-images \
  --dataset "../test_dataset" \
  --model-dir "../dataset" \
  --set-threshold
Step 4: Run the Server
python app.py
Server runs on http://localhost:3000

Using the API
Python Client
import requests

# Upload image
with open("image.jpg", "rb") as f:
    files = {"media": f}
    response = requests.post("http://localhost:3000/analyze", files=files)
    
result = response.json()
print(f"Result: {result['result']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Fake probability: {result['fake_probability']:.2%}")
JavaScript/Frontend
const formData = new FormData();
formData.append("media", fileInput.files[0]);

const response = await fetch("http://localhost:3000/analyze", {
  method: "POST",
  body: formData
});

const result = await response.json();
console.log(`Prediction: ${result.result}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
cURL
curl -X POST \
  -F "media=@image.jpg" \
  http://localhost:3000/analyze
Key Improvements Over v2.0
Feature	v2.0	v3.0
Backbone	MobileNetV3	ResNet50 ✨
Data Augmentation	Basic	Advanced ✨
Adversarial Training	FGSM	PGD ✨
Calibration	Adam	LBFGS ✨
Uncertainty Est.	MC Dropout	MC Dropout (25 passes) ✨
Video Model	CN	
NLSTM	ResNet+LSTM ✨	
Expected Accuracy	~92%	~96% ✨
Video Training (Optional)
If you have FaceForensics++ data:

python train.py train-video \
  --ffpp "../ffpp_data" \
  --epochs 8 \
  --batch 4 \
  --frames 12 \
  --adv-train
This creates video_cnn_lstm.pt for temporal deepfake detection.

Troubleshooting
"Model shows real/fake for only some media"
Solution: Train longer or with more data

# Try 20 epochs instead of 12
python train.py train-image --epochs 20 --adv-train

# Or increase batch size if you have GPU memory
python train.py train-image --batch 32 --epochs 15
"GPU out of memory"
# Use smaller batch size
python train.py train-image --batch 8

# Or use smaller model
python train.py train-image --arch resnet50 --batch 16
"Predictions always ~50% confidence"
Check if training converged:

cat dataset/metrics.json  # Look at "ece" - should be < 0.05
Retrain with more epochs:

python train.py train-image --epochs 20 --adv-train
"Server fails to start"
Check that model exists:

ls -la dataset/media_detector.pt  # Should be > 50MB
If missing, train first:

python train.py train-image --dataset "../dataset"
Production Deployment
1. Quantize Model (4x smaller, faster)
python train.py quantize --arch resnet50 \
  --in-model dataset/media_detector.pt \
  --out-model dataset/media_detector_int8.pt
2. Cross-Dataset Validation (Prove Generalization)
python train.py cross-dataset \
  --train-on "path/to/faceforensics" \
  --test-on "path/to/celeb_df" \
  --epochs 10
This proves your model works on unseen generation techniques.

3. Check Health Endpoint
curl http://localhost:3000/health

# Expected response:
# {"ok": true, "model": "media_detector.pt", "version": "3.0.0 (ResNet50)"}
Next Steps
Collect more diverse training data: Different cameras, lighting, compression levels
Test on cross-dataset: Validate on Celeb-DF or DFDC if available
Deploy with quantization: Use INT8 model for faster inference
Monitor predictions: Track confidence distributions over time
A/B test: Compare v2.0 vs v3.0 on your test set
Documentation
Full Upgrade Notes: See UPGRADE_NOTES.md
Architecture Details: See copilot-instructions.md
Backend README: See backend/README_BACKEND.md
Support
For issues:

Check error logs in terminal
Verify dataset structure
Try retraining with --adv-train flag
Test with smaller batch size if OOM
Good luck! 🎯
