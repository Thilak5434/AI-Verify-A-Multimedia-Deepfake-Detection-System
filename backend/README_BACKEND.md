## Run backend

```powershell
cd "c:\Users\Thilak\Downloads\main project\backend"
pip install -r requirements.txt
python app.py
```

Your existing frontend already calls `http://localhost:3000/analyze`.

## Train image model + calibration

```powershell
cd "c:\Users\Thilak\Downloads\main project\backend"
python train.py train-image --dataset "..\dataset" --arch mobilenet --epochs 8 --adv-train --out "..\dataset\media_detector.pt" --temp-out "..\dataset\temperature.json" --metrics-out "..\dataset\metrics.json"
```

## Cross-dataset evaluation

```powershell
python train.py cross-dataset --train-on "..\dataset" --test-on "..\dataset" --adv-train --cross-out "..\dataset\cross_dataset_metrics.json"
```

Replace `--test-on` with your local `Celeb-DF` or `DFDC` extracted image folders when available.

## Video branch (CNN+LSTM)

```powershell
python train.py train-video --ffpp "..\ffpp_data" --epochs 4 --video-out "..\dataset\video_cnn_lstm.pt"
```

## INT8 quantization

```powershell
python train.py quantize --in-model "..\dataset\media_detector.pt" --out-model "..\dataset\media_detector_int8.pt"
```
