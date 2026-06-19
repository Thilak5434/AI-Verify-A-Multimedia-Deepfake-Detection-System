# ============================================
# TECH STACK: PyTorch, NumPy, scikit-learn
# ============================================

# LIBRARIES
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

# LIBRARY: NumPy - Numerical operations
import numpy as np

# LIBRARY: PyTorch - Deep learning and data loading  
import torch
from torch.utils.data import DataLoader, Subset, random_split

# CORE TRAINING MODULES (from core.py)
from core import (
    AdvancedTrainer,           # TRAINING: Custom trainer with advanced features
    AudioClassifier,           # AUDIO MODEL: Audio classification network
    AudioDataset,              # AUDIO DATA: Audio file dataset loader
    BinaryImageFolder,         # IMAGE DATA: Image classification dataset
    CNNLSTM,                   # VIDEO MODEL: CNN-LSTM temporal model
    FFPPVideoDataset,          # VIDEO DATA: Video dataset from FFPP format
    ResNetBackbone,            # IMAGE MODEL: ResNet50/ResNet152/ResNet101 backbone
    build_inference_pack,      # FUNCTION: Build model for inference
    mc_dropout_predict,        # ALGORITHM: Monte Carlo Dropout for uncertainty
    eval_tf,                   # FUNCTION: Evaluation transforms (preprocessing)
    seed_all,                  # FUNCTION: Set random seeds for reproducibility
    train_tf,                  # FUNCTION: Training transforms (augmentation)
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# FILE PURPOSE: Command-line trainer
# This file trains detection models for:
#   - IMAGE: ResNet50/ResNet152 with MC Dropout
#   - VIDEO: CNN-LSTM temporal model
#   - AUDIO: AudioClassifier
# FEATURES: Cross-dataset evaluation, quantization (INT8), temperature scaling
# ============================================
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset"
DEFAULT_FFPP_DIR = ROOT_DIR / "ffpp_data"
DEFAULT_MEDIA_MODEL = DEFAULT_DATASET_DIR / "media_detector.pt"
DEFAULT_TEMP = DEFAULT_DATASET_DIR / "temperature.json"
DEFAULT_METRICS = DEFAULT_DATASET_DIR / "metrics.json"
DEFAULT_CROSS_METRICS = DEFAULT_DATASET_DIR / "cross_dataset_metrics.json"
DEFAULT_VIDEO_MODEL = DEFAULT_DATASET_DIR / "video_cnn_lstm.pt"
DEFAULT_VIDEO_METRICS = DEFAULT_DATASET_DIR / "video_metrics.json"
DEFAULT_VIDEO_CLASS_INDICES = DEFAULT_DATASET_DIR / "video_class_indices.json"
DEFAULT_INT8_MODEL = DEFAULT_DATASET_DIR / "media_detector_int8.pt"
DEFAULT_AUDIO_MODEL = DEFAULT_DATASET_DIR / "audio_classifier.pt"
DEFAULT_AUDIO_METRICS = DEFAULT_DATASET_DIR / "audio_metrics.json"
DEFAULT_AUDIO_DIR = ROOT_DIR / "AUDIO"


def _add_common_train_args(p, epochs=5, batch=16):
    # Shared options used by multiple subcommands.
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch", type=int, default=batch)
    p.add_argument("--lr", type=float, default=2e-4)  # Increased slightly for ResNet
    p.add_argument("--adv-train", action="store_true")
    p.add_argument("--warmup", type=int, default=500)  # Warmup for large models


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on validation split."""
    best_f1, best_th = -1.0, 0.5
    for th in np.linspace(0.2, 0.8, 61):
        pred = (p >= th).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th


def _predict_probs_with_pack(pack, dl) -> np.ndarray:
    """Return fake probabilities using the same model/scaler as inference."""
    probs = []
    pack.model.eval()
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(pack.device)
            m, _ = mc_dropout_predict(pack.model, x, n=10, scaler=pack.scaler)
            probs += m[:, 1].tolist()
    return np.asarray(probs, dtype=float)


def _class_weights_for_dataset(ds) -> Optional[torch.Tensor]:
    """Compute inverse-frequency class weights for (real=0, fake=1)."""
    base = ds.dataset if isinstance(ds, Subset) else ds
    idxs = ds.indices if isinstance(ds, Subset) else None

    labels: List[int] = []
    if hasattr(base, "items"):
        items = base.items
        if idxs is None:
            labels = [int(y) for _, y in items]
        else:
            labels = [int(items[i][1]) for i in idxs]
    else:
        # Fallback: iterate and read labels directly (may be slower).
        if idxs is None:
            for _, y in base:
                labels.append(int(y))
        else:
            for i in idxs:
                _, y = base[i]
                labels.append(int(y))

    c0 = sum(1 for y in labels if y == 0)
    c1 = sum(1 for y in labels if y == 1)
    if c0 == 0 or c1 == 0:
        logger.warning("Only one class present in dataset; skipping class-weighted loss.")
        return None

    total = c0 + c1
    w0 = total / (2 * c0)
    w1 = total / (2 * c1)
    return torch.tensor([w0, w1], dtype=torch.float32)

def run_eval_images(a):
    """Evaluate current model on a labeled image dataset and optionally tune threshold."""
    ds = BinaryImageFolder(a.dataset, tfm=eval_tf())
    if len(ds) < 20:
        raise RuntimeError(f"Dataset too small: {a.dataset}")

    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    pack = build_inference_pack(model_dir=a.model_dir, arch=a.arch)

    ys = np.asarray([y for _, y in ds], dtype=int)
    ps = _predict_probs_with_pack(pack, dl)
    th = _best_f1_threshold(ys, ps)
    met = AdvancedTrainer(pack.model, device=pack.device).evaluate(dl, threshold=th, calibrate=pack.scaler)

    if a.set_threshold:
        Path(a.model_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(a.model_dir) / "threshold.json").write_text(json.dumps({"threshold": th}, indent=2))

    out = {
        "dataset": a.dataset,
        "threshold": th,
        "metrics": met,
    }
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def run_train_image(a):
    # Fixed seed = reproducible experiments.
    seed_all(42)

    # Load dataset and create splits
    ds_len = BinaryImageFolder(a.dataset, tfm=eval_tf())
    if len(ds_len) < 50:
        logger.warning(f"Small dataset ({len(ds_len)} images). Results may not generalize well.")

    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(ds_len), generator=g).tolist()
    n_train = int(len(idx) * 0.8)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    tr_ds = Subset(BinaryImageFolder(a.dataset, tfm=train_tf()), tr_idx)
    va_ds = Subset(BinaryImageFolder(a.dataset, tfm=eval_tf()), va_idx)
    
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    # Create ResNet model
    model = ResNetBackbone(arch=a.arch, dropout=0.35)
    class_weights = _class_weights_for_dataset(tr_ds)
    trainer = AdvancedTrainer(model, class_weights=class_weights)
    
    # Optimizer with weight decay for better generalization
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best_f1, best_state = -1.0, None
    logger.info(f"Training ResNet {a.arch} for {a.epochs} epochs with {len(tr_ds)} images")
    
    for ep in range(a.epochs):
        loss = trainer.train_epoch(tr_dl, opt, use_adv=a.adv_train)
        met = trainer.evaluate(va_dl)
        scheduler.step()
        
        if met["f1"] > best_f1:
            best_f1, best_state = met["f1"], {k: v.cpu() for k, v in model.state_dict().items()}
        
        logger.info(f"epoch={ep+1} loss={loss:.4f} acc={met['accuracy']:.4f} f1={met['f1']:.4f} auc={met['auc']:.4f} ece={met['ece']:.4f}")

    # Save best model
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), a.out)
    logger.info(f"Saved model to {a.out}")

    # Calibrate temperature
    model.load_state_dict(torch.load(a.out, map_location=trainer.device), strict=True)
    scaler, temp = trainer.fit_temperature(va_dl)
    Path(a.temp_out).write_text(json.dumps({"temperature": temp}, indent=2))
    logger.info(f"Temperature scaling: {temp:.4f}")

    # Compute best threshold
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for x, y in va_dl:
            logits = model(x.to(trainer.device))
            if scaler is not None:
                logits = scaler(logits)
            ys += y.numpy().tolist()
            ps += torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
    
    y, p = np.asarray(ys, dtype=int), np.asarray(ps, dtype=float)
    th = _best_f1_threshold(y, p)
    Path(str(Path(a.out).parent / "threshold.json")).write_text(json.dumps({"threshold": th}, indent=2))
    logger.info(f"Optimal threshold: {th:.4f}")

    # Save metrics
    uncal = trainer.evaluate(va_dl, threshold=th)
    cal = trainer.evaluate(va_dl, threshold=th, calibrate=scaler)
    payload = {
        "model_arch": a.arch,
        "val_uncalibrated": uncal,
        "val_calibrated": cal,
        "threshold": th,
        "temperature": temp,
        "notes": "ResNet-based detector with temperature scaling and PGD adversarial training",
    }
    Path(a.metrics_out).write_text(json.dumps(payload, indent=2))
    logger.info("Training complete!")
    print(f"saved: {a.out}, {a.temp_out}, {a.metrics_out}")


def run_cross_dataset(a):
    # Train on one dataset root and evaluate on another dataset root.
    seed_all(42)
    for p in [a.train_on, a.test_on]:
        if not Path(p).exists():
            raise RuntimeError(f"Missing dataset path: {p}")

    tr_ds = BinaryImageFolder(a.train_on, tfm=train_tf())
    te_ds = BinaryImageFolder(a.test_on, tfm=eval_tf())
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    model = ResNetBackbone(arch=a.arch)
    class_weights = _class_weights_for_dataset(tr_ds)
    trainer = AdvancedTrainer(model, class_weights=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    logger.info(f"Cross-dataset: Train on {a.train_on}, Test on {a.test_on}")
    for ep in range(a.epochs):
        loss = trainer.train_epoch(tr_dl, opt, use_adv=a.adv_train)
        scheduler.step()
        logger.info(f"epoch={ep+1} loss={loss:.4f}")

    # Save a single JSON file so cross-dataset runs are easy to track.
    met = trainer.evaluate(te_dl)
    Path(a.cross_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.cross_out).write_text(
        json.dumps(
            {
                "train_on": a.train_on,
                "test_on": a.test_on,
                "model_arch": a.arch,
                "metrics": met,
                "note": "For strong claims: train on FF++, test on Celeb-DF/DFDC.",
            },
            indent=2,
        )
    )
    logger.info(f"cross-dataset metrics saved: {a.cross_out}")
    print(json.dumps(met, indent=2))


def run_train_video(a):
    # Video loader now supports old and new folder layouts.
    seed_all(42)
    ds = FFPPVideoDataset(a.ffpp, frames_per_video=a.frames, max_videos=a.max_videos)
    if len(ds) < 4:
        raise RuntimeError("Not enough videos in FF++ folder.")

    n_train = int(len(ds) * 0.8)
    n_val = max(1, len(ds) - n_train)
    tr_ds, va_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    model = CNNLSTM(hidden=256)
    class_weights = _class_weights_for_dataset(tr_ds)
    trainer = AdvancedTrainer(model, class_weights=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best_f1 = -1.0
    logger.info(f"Training CNNLSTM with {len(tr_ds)} videos")
    
    for ep in range(a.epochs):
        loss = trainer.train_epoch(tr_dl, opt, use_adv=a.adv_train)
        met = trainer.evaluate(va_dl)
        scheduler.step()
        best_f1 = max(best_f1, met["f1"])
        logger.info(f"video-epoch={ep+1} loss={loss:.4f} acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

    Path(a.video_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.video_out)
    
    payload = {
        "model": "CNNLSTM",
        "frames_per_video": a.frames,
        "best_f1": best_f1,
        "val_metrics": met,
    }
    Path(a.video_metrics_out).write_text(json.dumps(payload, indent=2))
    Path(a.video_class_indices_out).write_text(json.dumps({"real": 0, "fake": 1}, indent=2))
    logger.info(f"saved: {a.video_out}")
    print(json.dumps(payload, indent=2))


def run_quantize(a):
    """INT8 dynamic quantization for faster CPU inference."""
    model = ResNetBackbone(arch=a.arch)
    model.load_state_dict(torch.load(a.in_model, map_location="cpu"), strict=False)
    q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(q.state_dict(), a.out_model)
    logger.info(f"quantized model saved: {a.out_model}")
    print(f"Quantization complete: {a.out_model}")


def run_train_audio(a):
    """Train audio classifier for voice deepfake detection."""
    from sklearn.model_selection import train_test_split
    
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training audio classifier on {device}")
    
    # Load audio dataset
    ds = AudioDataset(a.audio_path, sr=16000, duration=5.0)
    if len(ds) == 0:
        raise ValueError(f"No audio files found in {a.audio_path}")
    
    # Extract labels for stratified split
    labels = [y for _, y in ds.items]
    logger.info(f"Dataset size: {len(ds)}")
    logger.info(f"  REAL samples: {sum(1 for y in labels if y == 0)}")
    logger.info(f"  FAKE samples: {sum(1 for y in labels if y == 1)}")
    
    # Use STRATIFIED split to ensure both classes in train and val
    train_idx, val_idx = train_test_split(
        range(len(ds)), 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    tr_ds = Subset(ds, train_idx)
    va_ds = Subset(ds, val_idx)
    
    # Verify both classes exist in splits
    tr_labels = [labels[i] for i in train_idx]
    va_labels = [labels[i] for i in val_idx]
    tr_real = sum(1 for y in tr_labels if y == 0)
    tr_fake = sum(1 for y in tr_labels if y == 1)
    va_real = sum(1 for y in va_labels if y == 0)
    va_fake = sum(1 for y in va_labels if y == 1)
    
    logger.info(f"✓ Train split: {tr_real} REAL, {tr_fake} FAKE")
    logger.info(f"✓ Val split:   {va_real} REAL, {va_fake} FAKE")
    
    if tr_real == 0 or tr_fake == 0 or va_real == 0 or va_fake == 0:
        raise ValueError("ERROR: Stratified split failed - one class missing in train or val!")
def run_train_audio(a):
    """Train audio classifier for voice deepfake detection with improved convergence."""
    from sklearn.model_selection import train_test_split
    
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training audio classifier on {device}")
    
    # Load audio dataset
    ds = AudioDataset(a.audio_path, sr=16000, duration=5.0)
    if len(ds) == 0:
        raise ValueError(f"No audio files found in {a.audio_path}")
    
    # Extract labels for stratified split
    labels = [y for _, y in ds.items]
    logger.info(f"Dataset size: {len(ds)}")
    logger.info(f"  REAL samples: {sum(1 for y in labels if y == 0)}")
    logger.info(f"  FAKE samples: {sum(1 for y in labels if y == 1)}")
    
    # Use STRATIFIED split to ensure both classes in train and val
    train_idx, val_idx = train_test_split(
        range(len(ds)), 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    tr_ds = Subset(ds, train_idx)
    va_ds = Subset(ds, val_idx)
    
    # Verify both classes exist in splits
    tr_labels = [labels[i] for i in train_idx]
    va_labels = [labels[i] for i in val_idx]
    tr_real = sum(1 for y in tr_labels if y == 0)
    tr_fake = sum(1 for y in tr_labels if y == 1)
    va_real = sum(1 for y in va_labels if y == 0)
    va_fake = sum(1 for y in va_labels if y == 1)
    
    logger.info(f"✓ Train split: {tr_real} REAL, {tr_fake} FAKE")
    logger.info(f"✓ Val split:   {va_real} REAL, {va_fake} FAKE")
    
    if tr_real == 0 or tr_fake == 0 or va_real == 0 or va_fake == 0:
        raise ValueError("ERROR: Stratified split failed - one class missing in train or val!")
    
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=a.batch, shuffle=False, num_workers=0)
    
    # Build model with moderate regularization (not too aggressive)
    model = AudioClassifier(dropout=0.3)  # Reduced from 0.5 - too aggressive for small dataset
    class_weights = _class_weights_for_dataset(tr_ds)
    trainer = AdvancedTrainer(model, device=device, class_weights=class_weights)
    
    # Use higher learning rate for better convergence on small dataset
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)  # Increased LR & weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4, T_mult=1, eta_min=1e-5)
    
    best_f1 = -1.0
    best_model_state = None
    logger.info(f"Training AudioClassifier with {len(tr_ds)} samples, batch_size={a.batch}")
    
    for ep in range(a.epochs):
        loss = trainer.train_epoch(tr_dl, opt, use_adv=a.adv_train)
        met = trainer.evaluate(va_dl)
        scheduler.step()
        
        if met["f1"] > best_f1:
            best_f1 = met["f1"]
            best_model_state = model.state_dict().copy()
        
        logger.info(f"audio-epoch={ep+1}/{a.epochs} loss={loss:.4f} acc={met['accuracy']:.4f} f1={met['f1']:.4f} auc={met['auc']:.4f}")
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"✓ Loaded best model (F1={best_f1:.4f})")
    
    Path(a.audio_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.audio_out)
    
    # Final evaluation with best model
    met = trainer.evaluate(va_dl)
    
    payload = {
        "model": "AudioClassifier",
        "duration": 5.0,
        "sr": 16000,
        "dropout": 0.3,  # Updated to match actual dropout
        "best_f1": best_f1,
        "val_metrics": met,
    }
    Path(a.audio_metrics_out).write_text(json.dumps(payload, indent=2))
    logger.info(f"✓ saved: {a.audio_out}")
    logger.info(f"✓ Final metrics - Accuracy: {met['accuracy']:.4f}, F1: {met['f1']:.4f}, AUC: {met['auc']:.4f}")
    print(json.dumps(payload, indent=2))


def main():
    # Main CLI parser with subcommands for different tasks.
    p = argparse.ArgumentParser(description="Deepfake detector training/evaluation CLI (ResNet-based)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train image classifier
    s1 = sub.add_parser("train-image", help="Train image deepfake detector")
    s1.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Path to dataset with real/ and fake/ folders")
    s1.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet101", "resnet152"], help="Model architecture")
    _add_common_train_args(s1, epochs=12, batch=16)
    s1.add_argument("--out", default=str(DEFAULT_MEDIA_MODEL), help="Output model path")
    s1.add_argument("--temp-out", default=str(DEFAULT_TEMP), help="Temperature calibration output")
    s1.add_argument("--metrics-out", default=str(DEFAULT_METRICS), help="Metrics output")

    # Cross-dataset evaluation
    s2 = sub.add_parser("cross-dataset", help="Train on one dataset, test on another")
    s2.add_argument("--train-on", required=True, help="Training dataset path")
    s2.add_argument("--test-on", required=True, help="Test dataset path")
    s2.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet101", "resnet152"])
    _add_common_train_args(s2, epochs=10, batch=16)
    s2.add_argument("--cross-out", default=str(DEFAULT_CROSS_METRICS))

    # Train video model
    s3 = sub.add_parser("train-video", help="Train CNNLSTM video detector")
    s3.add_argument("--ffpp", default=str(DEFAULT_FFPP_DIR), help="FaceForensics++ dataset path")
    _add_common_train_args(s3, epochs=8, batch=4)
    s3.add_argument("--frames", type=int, default=12, help="Frames per video")
    s3.add_argument("--max-videos", type=int, default=200, help="Max videos to use (0=all)")
    s3.add_argument("--video-out", default=str(DEFAULT_VIDEO_MODEL))
    s3.add_argument("--video-metrics-out", default=str(DEFAULT_VIDEO_METRICS))
    s3.add_argument("--video-class-indices-out", default=str(DEFAULT_VIDEO_CLASS_INDICES))

    # Quantize model
    s4 = sub.add_parser("quantize", help="Quantize model to INT8")
    s4.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet101", "resnet152"])
    s4.add_argument("--in-model", default=str(DEFAULT_MEDIA_MODEL), help="Input model")
    s4.add_argument("--out-model", default=str(DEFAULT_INT8_MODEL), help="Output quantized model")

    # Evaluate on dataset
    s5 = sub.add_parser("eval-images", help="Evaluate on image dataset")
    s5.add_argument("--dataset", required=True)
    s5.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet101", "resnet152"])
    s5.add_argument("--model-dir", default=str(DEFAULT_DATASET_DIR))
    s5.add_argument("--batch", type=int, default=32)
    s5.add_argument("--set-threshold", action="store_true")
    s5.add_argument("--out", default="")

    # Train audio model
    s6 = sub.add_parser("train-audio", help="Train audio deepfake detector")
    s6.add_argument("--audio-path", default=str(DEFAULT_AUDIO_DIR), help="Path to audio dataset with real/ and fake/ folders")
    _add_common_train_args(s6, epochs=8, batch=16)
    s6.add_argument("--audio-out", default=str(DEFAULT_AUDIO_MODEL), help="Output audio model path")
    s6.add_argument("--audio-metrics-out", default=str(DEFAULT_AUDIO_METRICS), help="Audio metrics output")

    a = p.parse_args()
    if a.cmd == "train-image":
        run_train_image(a)
    elif a.cmd == "cross-dataset":
        run_cross_dataset(a)
    elif a.cmd == "train-video":
        run_train_video(a)
    elif a.cmd == "quantize":
        run_quantize(a)
    elif a.cmd == "eval-images":
        run_eval_images(a)
    elif a.cmd == "train-audio":
        run_train_audio(a)


if __name__ == "__main__":
    main()
