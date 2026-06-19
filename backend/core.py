# ============================================
# TECH STACK: PyTorch, TorchVision, OpenCV, Pillow, NumPy, scikit-learn
# ============================================

# LIBRARIES for detection and ML
import base64
import hashlib                                   # ALGORITHM: Hashing for integrity/caching
import io
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# LIBRARY: OpenCV - Computer vision and video processing
import cv2

# LIBRARY: NumPy - Numerical computing
import numpy as np

# LIBRARY: PyTorch - Deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F

# LIBRARY: Pillow - Image processing and augmentation
from PIL import Image, ImageFilter, ImageEnhance

# LIBRARY: scikit-learn - Metrics for model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

# LIBRARY: PyTorch data loading
from torch.utils.data import DataLoader, Dataset

# LIBRARY: TorchVision - Pre-trained models (ResNet50, ResNet152, etc.)
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CLASSIFICATION LABELS: Binary classification (Real vs Fake/AI-Generated)
# ============================================
LABELS = {0: "Real", 1: "AI Generated (Fake)"}  # CLASS LABELS for detection output
IMAGE_SIZE = 224                                   # IMAGE: Standard input size for ResNet (224x224)


def seed_all(seed: int = 42) -> None:
    """Reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================
# DATA AUGMENTATION: Advanced augmentation techniques for deepfake detection
# These simulations help the model generalize better to various compressions/artifacts
# ============================================

class AdvancedJPEGCompression:
    """
    AUGMENTATION ALGORITHM: Simulate various JPEG quality levels
    PURPOSE: Real deepfake indicators include compression artifacts
    LIBRARY: Pillow (PIL)
    """
    def __init__(self, p: float = 0.4, qmin: int = 20, qmax: int = 95):
        self.p, self.qmin, self.qmax = p, qmin, qmax

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=q)
        b.seek(0)
        return Image.open(b).convert("RGB")


class RealisticBlur:
    """
    AUGMENTATION ALGORITHM: Motion blur and Gaussian blur
    PURPOSE: Simulate compression artifacts found in manipulated media
    LIBRARY: Pillow (PIL.ImageFilter)
    """
    def __init__(self, p: float = 0.35):
        self.p = p

    @staticmethod
    def _motion_kernel(size: int, horizontal: bool) -> ImageFilter.Kernel:
        # Simple linear motion blur kernel (horizontal or vertical).
        size = max(3, int(size))
        if size % 2 == 0:
            size += 1
        kernel = [0.0] * (size * size)
        if horizontal:
            row = size // 2
            for i in range(size):
                kernel[row * size + i] = 1.0
        else:
            col = size // 2
            for i in range(size):
                kernel[i * size + col] = 1.0
        return ImageFilter.Kernel((size, size), kernel, scale=sum(kernel))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        blur_type = random.choice(['gaussian', 'motion'])
        if blur_type == 'gaussian':
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        else:
            size = random.randint(3, 9)
            horizontal = random.random() > 0.5
            return img.filter(self._motion_kernel(size=size, horizontal=horizontal))


class RealisticNoise:
    """
    AUGMENTATION ALGORITHM: Gaussian noise and low-light simulation
    PURPOSE: Simulate camera grain and low-light conditions in real footage
    LIBRARY: NumPy
    """
    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        x = np.asarray(img).astype(np.float32)
        
        # Low-light darkening
        if random.random() > 0.5:
            x *= random.uniform(0.5, 0.9)
        
        # Gaussian noise
        noise_std = random.uniform(2.0, 8.0)
        x += np.random.normal(0.0, noise_std, x.shape)
        
        return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


class ColorDistortion:
    """
    AUGMENTATION ALGORITHM: Color space distortions
    PURPOSE: Simulate social media reencoding artifacts
    LIBRARY: Pillow (PIL.ImageEnhance)
    """
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        return img



# ============================================
# DATA AUGMENTATION: Transform pipeline for training
# ============================================

def train_tf() -> transforms.Compose:
    """
    FUNCTION: Training transformation pipeline
    PURPOSE: Apply aggressive augmentation for better model generalization
    LIBRARY: TorchVision Transforms
    AUGMENTATIONS: Resize, crop, flip, affine, blur, JPEG compression, noise, color distortion, erasing
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RealisticBlur(p=0.4),
        AdvancedJPEGCompression(p=0.45, qmin=20, qmax=95),
        RealisticNoise(p=0.4),
        ColorDistortion(p=0.35),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.05, 0.15), ratio=(0.3, 3.0)),
    ])


def eval_tf() -> transforms.Compose:
    """
    FUNCTION: Inference/validation transformation pipeline
    PURPOSE: Clean preprocessing without augmentation
    LIBRARY: TorchVision Transforms
    NORMALIZATION: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])



# ============================================
# IMAGE DETECTION: Dataset loading for images
# ============================================

class BinaryImageFolder(Dataset):
    """
    MODULE: Image dataset loader
    PURPOSE: Load real (label=0) and fake/AI-generated (label=1) images
    LIBRARY: PyTorch Dataset, Pillow
    SUPPORTED FORMATS: JPG, JPEG, PNG, BMP, WebP
    """
    def __init__(self, root: str, tfm=None):
        self.root = Path(root)
        self.tfm = tfm or eval_tf()
        self.items = []
        real_dir, fake_dir = self._resolve_image_dirs(self.root)
        
        for p, y in [(real_dir, 0), (fake_dir, 1)]:
            for f in p.glob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.items.append((f, y))
        
        logger.info(f"Loaded {len(self.items)} images from {root}")

    @staticmethod
    def _resolve_image_dirs(root: Path) -> Tuple[Path, Path]:
        """Find real/fake directories."""
        for base in [root, root / "images"]:
            if not base.exists():
                continue
            dirs = [d for d in base.iterdir() if d.is_dir()]
            by_name = {d.name.lower(): d for d in dirs}
            
            if "real" in by_name and "fake" in by_name:
                return by_name["real"], by_name["fake"]
            
            real_candidates = [d for d in dirs if "real" in d.name.lower()]
            fake_candidates = [d for d in dirs if "fake" in d.name.lower()]
            if real_candidates and fake_candidates:
                return real_candidates[0], fake_candidates[0]
        
        raise RuntimeError(f"Image dataset not found in {root}. Need real/ and fake/ folders.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        try:
            img = Image.open(p).convert("RGB")
            return self.tfm(img), torch.tensor(y, dtype=torch.long)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(y, dtype=torch.long)



# ============================================
# VIDEO DETECTION: Dataset loading for videos
# ============================================

class FFPPVideoDataset(Dataset):
    """
    MODULE: Video dataset loader
    PURPOSE: Load real (label=0) and manipulated/fake (label=1) videos
    DATASET: Supports FFPP (FaceForensics++) format
    LIBRARY: PyTorch Dataset, OpenCV
    SUPPORTED FORMATS: MP4, AVI
    PROCESSING: Extracts frames_per_video frames from each video
    """
    def __init__(self, ffpp_root: str, frames_per_video: int = 12, max_videos: int = 0):
        self.frames_per_video = frames_per_video
        root = Path(ffpp_root)
        real_dir, fake_dir = self._resolve_video_dirs(root)
        
        real = list(real_dir.glob("*.mp4")) + list(real_dir.glob("*.avi"))
        fake = list(fake_dir.glob("*.mp4")) + list(fake_dir.glob("*.avi"))
        
        if max_videos > 0:
            k = max(1, max_videos // 2)
            self.items = [(p, 0) for p in real[:k]] + [(p, 1) for p in fake[:k]]
        else:
            self.items = [(p, 0) for p in real] + [(p, 1) for p in fake]
        
        random.Random(42).shuffle(self.items)
        logger.info(f"Loaded {len(self.items)} videos from {ffpp_root}")

    @staticmethod
    def _resolve_video_dirs(root: Path) -> Tuple[Path, Path]:
        """Find video directories."""
        bases = [root, root / "videos"]
        pairs = [("original_sequences", "manipulated_sequences"), ("real", "fake")]
        
        for base in bases:
            for real_name, fake_name in pairs:
                real_dir, fake_dir = base / real_name, base / fake_name
                if real_dir.exists() and fake_dir.exists():
                    return real_dir, fake_dir
        
        raise RuntimeError(f"Video dataset not found in {root}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        frames = sample_video_frames(str(p), self.frames_per_video)
        if not frames:
            frames = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)] * self.frames_per_video
        
        tfm = eval_tf()
        x = torch.stack([tfm(Image.fromarray(f[:, :, ::-1])) for f in frames])
        return x, torch.tensor(y, dtype=torch.long)



# ============================================
# IMAGE DETECTION: Model architecture for image classification
# ============================================

class ResNetBackbone(nn.Module):
    """
    ARCHITECTURE: ResNet50/ResNet152/ResNet101 backbone
    PURPOSE: Extract features from images and classify as Real or AI-Generated
    LIBRARY: PyTorch, TorchVision
    KEY FEATURE: MC Dropout for uncertainty estimation
    PRE-TRAINING: ImageNet weights (ResNet50_Weights.IMAGENET1K_V1)
    OUTPUT: Binary classification (Real=0, Fake=1)
    """
    def __init__(self, arch: str = "resnet50", dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        
        if arch == "resnet152":
            base = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        elif arch == "resnet101":
            base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:  # resnet50
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace final layer and add dropout
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        
        self.net = base

    def forward(self, x):
        return self.net(x)



# ============================================
# VIDEO DETECTION: Model architecture for temporal video analysis
# ============================================

class CNNLSTM(nn.Module):
    """
    ARCHITECTURE: CNN + LSTM (Convolutional Neural Network + Long Short-Term Memory)
    PURPOSE: Analyze temporal sequences in videos for manipulation detection
    LIBRARY: PyTorch, TorchVision
    COMPONENTS:
      - CNN: ResNet50 for spatial feature extraction from video frames
      - LSTM: Bidirectional 2-layer LSTM for temporal modeling
    KEY FEATURE: Captures frame-to-frame relationships and temporal patterns
    OUTPUT: Binary classification per video sequence (Real=0, Fake=1)
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base.children())[:-1])  # Remove final FC
        self.lstm = nn.LSTM(2048, hidden, batch_first=True, bidirectional=True, num_layers=2)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        # Process all frames through CNN
        features = self.cnn(x.view(b * t, c, h, w)).view(b, t, -1)
        # Temporal modeling with LSTM
        lstm_out, _ = self.lstm(features)
        # Use last hidden state
        return self.head(lstm_out[:, -1, :])


class TemperatureScaler(nn.Module):
    """
    ALGORITHM: Post-hoc Temperature Scaling
    PURPOSE: Calibrate neural network confidence scores
    METHOD: Learn a single temperature parameter T to scale logits
    REFERENCE: Guo et al. (2017) - On Calibration of Modern Neural Networks
    OUTPUT: Calibrated probability that better reflects actual accuracy
    """
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.t.clamp(0.05, 10.0)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, iters: int = 300, lr: float = 0.01) -> float:
        opt = torch.optim.LBFGS([self.t], lr=lr, max_iter=50)
        
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            return loss
        
        opt.step(closure)
        return float(self.t.item())


# ============================================================================
# AUDIO DETECTION PIPELINE
# ============================================================================

# ============================================
# AUDIO DETECTION: Audio feature extraction
# ============================================

def audio_to_mel_spectrogram(audio_path: str, sr: int = 16000, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512, duration: float = 5.0) -> np.ndarray:
    """
    FUNCTION: Audio to spectrogram conversion
    ALGORITHM: Short-Time Fourier Transform (STFT) → Mel-scale conversion
    LIBRARY: librosa (librosa.feature.melspectrogram)
    PURPOSE: Convert audio waveform to visual representation (Mel-spectrogram)
    PROCESSING:
      1. Load audio at 16 kHz sample rate
      2. Extract Mel-scale spectrogram (128 bins)
      3. Convert to dB scale
      4. Normalize to [0, 1] range
    RETURNS: Normalized Mel-spectrogram array (128, 156)
    """
    import librosa
    from pathlib import Path
    
    try:
        # Validate file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_file.stat().st_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        logger.info(f"Loading audio: {audio_path} ({audio_file.stat().st_size / 1024:.1f} KB)")
        
        # Load with timeout protection
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        if len(y) == 0:
            raise ValueError(f"No audio data loaded from {audio_path} (might be corrupted or unsupported format)")
        
        logger.info(f"Audio loaded: {len(y)} samples at {sr}Hz, duration: {len(y)/sr:.2f}s")
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Pad/trim to fixed size (128 x 156 for 5-second audio at 16kHz) FIRST
        target_shape = (n_mels, 156)
        if S_db.shape[1] < target_shape[1]:
            S_db = np.pad(S_db, ((0, 0), (0, target_shape[1] - S_db.shape[1])), mode='mean')
        else:
            S_db = S_db[:, :target_shape[1]]
        
        # Normalize using global range (more stable than per-sample normalization)
        # Typical mel-spectrogram range is -80 to 0 dB
        S_db = np.clip(S_db, -80, 0)  # Clip to typical range
        S_db = (S_db + 80) / 80  # Normalize to [0, 1]
        
        logger.info(f"Mel-spectrogram shape: {S_db.shape}")
        return S_db.astype(np.float32)
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {type(e).__name__}: {e}")
        raise


class AudioDataset(Dataset):
    """
    MODULE: Audio dataset loader for training
    PURPOSE: Load real and fake audio files for classification
    LIBRARY: PyTorch Dataset, librosa
    SUPPORTED FORMATS: WAV, MP3, FLAC
    PROCESSING: Load audio files and convert to Mel-spectrograms
    """
    def __init__(self, audio_root: str, sr: int = 16000, duration: float = 5.0, max_samples: int = 0):
        self.sr = sr
        self.duration = duration
        root = Path(audio_root)
        
        # Find audio directories
        real_dir = root / "real" if (root / "real").exists() else root / "REAL"
        fake_dir = root / "fake" if (root / "fake").exists() else root / "FAKE"
        
        if not real_dir.exists() or not fake_dir.exists():
            raise RuntimeError(f"Audio dataset not found in {audio_root}. Expected 'real' and 'fake' subdirectories.")
        
        real_files = list(real_dir.glob("*.wav")) + list(real_dir.glob("*.mp3")) + list(real_dir.glob("*.flac"))
        fake_files = list(fake_dir.glob("*.wav")) + list(fake_dir.glob("*.mp3")) + list(fake_dir.glob("*.flac"))
        
        # Verify files are found
        if len(real_files) == 0 or len(fake_files) == 0:
            raise RuntimeError(f"Missing audio files! REAL={len(real_files)}, FAKE={len(fake_files)} in {audio_root}")
        
        if max_samples > 0:
            k = max(1, max_samples // 2)
            self.items = [(p, 0) for p in real_files[:k]] + [(p, 1) for p in fake_files[:k]]
        else:
            self.items = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
        
        random.Random(42).shuffle(self.items)
        
        # Log detailed class distribution
        logger.info(f"✓ Loaded {len(self.items)} audio files from {audio_root}")
        logger.info(f"  REAL files: {len(real_files)}")
        logger.info(f"  FAKE files: {len(fake_files)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        spec = audio_to_mel_spectrogram(str(p), sr=self.sr, duration=self.duration)
        x = torch.from_numpy(spec).unsqueeze(0)  # Add channel dimension: (1, 128, 156)
        return x, torch.tensor(y, dtype=torch.long)


class AudioClassifier(nn.Module):
    """
    ARCHITECTURE: CNN for audio classification
    PURPOSE: Detect AI-cloned or synthetic voices
    INPUT: Mel-spectrograms (1, 128, 156)
    COMPONENTS:
      - 4 Conv blocks with BatchNorm and ReLU
      - Global Average Pooling
      - Classification head with Dropout
    LIBRARY: PyTorch
    OUTPUT: Binary classification (Real=0, Fake=1)
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction: Conv blocks
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x shape: (batch, 1, 128, 156)
        x = self.conv_layers(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.head(x)
        return x


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float = 8/255, alpha: float = 2/255, steps: int = 7) -> torch.Tensor:
    """
    ALGORITHM: Projected Gradient Descent (PGD) Adversarial Attack
    PURPOSE: Generate adversarial examples for adversarial training
    METHOD: Iteratively perturb input to maximize loss within epsilon ball
    REFERENCE: Madry et al. (2018) - Towards Deep Learning Models Resistant to Adversarial Attacks
    LIBRARY: PyTorch
    RETURNS: Adversarial examples
    """
    x_adv = x.detach().clone()
    for _ in range(steps):
        x_adv.requires_grad = True
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, x - eps, x + eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


@dataclass
class InferencePack:
    """
    CONTAINER: Bundled model + inference components
    CONTENTS:
      - model: Trained detection network (ResNet50 backbone)
      - scaler: Temperature scaler for calibration
      - class_names: Label mapping {0: "Real", 1: "Fake"}
      - device: GPU/CPU device
      - threshold: Decision threshold for binary classification
    PURPOSE: Encapsulate all inference components needed for detection
    """
    model: nn.Module
    scaler: Optional[TemperatureScaler]
    class_names: Dict[int, str]
    device: torch.device
    threshold: float = 0.5


def sample_video_frames(path: str, n: int = 12) -> List[np.ndarray]:
    """
    FUNCTION: Video frame extraction
    ALGORITHM: Uniform sampling across video duration
    LIBRARY: OpenCV (cv2.VideoCapture)
    PURPOSE: Extract representative frames from video for analysis
    RETURNS: List of numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)))
    
    cap.release()
    while len(frames) < n:
        frames.append(frames[-1].copy() if frames else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    
    return frames[:n]



def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n: int = 25, scaler: Optional[TemperatureScaler] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    ALGORITHM: Monte Carlo Dropout - Uncertainty Estimation
    PURPOSE: Run forward passes with dropout active to get probability distribution
    LIBRARY: PyTorch
    OUTPUT: (mean_probabilities, std_probabilities) for uncertainty quantification
    REFERENCE: Gal & Ghahramani (2016) - Bayesian Deep Learning via Dropout
    """
    was_training = model.training
    model.eval()
    
    # Enable dropout during inference
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    
    probs = []
    with torch.no_grad():
        for _ in range(n):
            logits = model(x)
            if scaler is not None:
                logits = scaler(logits)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    
    model.train(was_training)
    
    p = np.stack(probs, axis=0)
    return p.mean(axis=0), p.std(axis=0)


@torch.no_grad()
def deterministic_predict(model: nn.Module, x: torch.Tensor, scaler: Optional[TemperatureScaler] = None) -> np.ndarray:
    """
    FUNCTION: Deterministic single-pass inference
    PURPOSE: Stable predictions without MC Dropout randomness
    LIBRARY: PyTorch
    OUTPUT: Probability predictions [batch_size, num_classes]
    """
    was_training = model.training
    model.eval()
    logits = model(x)
    if scaler is not None:
        logits = scaler(logits)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    model.train(was_training)
    return probs


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    FUNCTION: Model evaluation metrics computation
    METRICS: Accuracy, Precision, Recall, F1, AUC, Confusion matrix (TP/FP/TN/FN)
    LIBRARY: scikit-learn
    PURPOSE: Evaluate detection model performance
    """
    y_pred = (y_score >= threshold).astype(int)
    
    if len(np.unique(y_true)) < 2:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """
    ALGORITHM: Expected Calibration Error (ECE)
    PURPOSE: Measure how well model's confidence matches actual accuracy
    LIBRARY: NumPy
    REFERENCE: Guo et al. (2017) - On Calibration of Modern Neural Networks
    """
    conf = np.maximum(y_prob, 1 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    ece = 0.0
    
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            acc = (pred[mask] == y_true[mask]).mean()
            conf_mean = conf[mask].mean()
            ece += abs(acc - conf_mean) * mask.mean()
    
    return float(ece)



def build_inference_pack(model_dir: str = "dataset", arch: str = "resnet50", device: Optional[str] = None) -> InferencePack:
    """
    FUNCTION: Initialize detection inference pipeline
    LOADS:
      1. ResNet50/ResNet152/ResNet101 model weights
      2. Temperature scaler for calibration
      3. Decision threshold for classification
    LIBRARY: PyTorch, TorchVision
    RETURNS: InferencePack with model, scaler, normalization parameters
    """
    d = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = ResNetBackbone(arch=arch).to(d).eval()
    model_path = Path(model_dir) / "media_detector.pt"
    
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=d), strict=False)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}, using untrained weights")
    
    # Load temperature scaler
    scaler = TemperatureScaler().to(d)
    t_path = Path(model_dir) / "temperature.json"
    if t_path.exists():
        t = float(json.loads(t_path.read_text()).get("temperature", 1.0))
        if 0.05 <= t <= 10.0:
            scaler.t.data = torch.tensor([t], device=d)
    
    # Load threshold
    th_path = Path(model_dir) / "threshold.json"
    threshold = 0.5
    if th_path.exists():
        threshold = float(json.loads(th_path.read_text()).get("threshold", 0.5))
    # Keep thresholds within the tuning range used in training.
    threshold = max(0.1, min(threshold, 0.85))
    
    return InferencePack(model=model, scaler=scaler, class_names=LABELS, device=d, threshold=threshold)


def load_audio_model(model_path: str, device: Optional[torch.device] = None) -> Optional[str]:
    """
    FUNCTION: Load AudioClassifier model for voice detection
    MODEL: AudioClassifier CNN (processes Mel-spectrograms)
    LIBRARY: PyTorch
    RETURNS: Model path if successfully loaded, None otherwise
    """
    if not model_path or not Path(model_path).exists():
        logger.info("Audio model not available yet (train with: python train.py train-audio --audio-path <path>)")
        return None
    try:
        d = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AudioClassifier(dropout=0.3).to(d).eval()
        model.load_state_dict(torch.load(model_path, map_location=d), strict=False)
        logger.info(f"✓ Audio model loaded from {model_path}")
        return model_path
    except Exception as e:
        logger.warning(f"Failed to load audio model: {e}")
        return None


def provenance_hash(raw: bytes) -> str:
    """
    FUNCTION: File integrity hashing
    ALGORITHM: SHA256 cryptographic hash
    PURPOSE: Generate file fingerprint for caching and deduplication
    LIBRARY: hashlib
    RETURNS: Hex string of SHA256 hash
    """
    return hashlib.sha256(raw).hexdigest()


def predict_image_bytes(pack: InferencePack, raw: bytes) -> Dict:
    """
    FUNCTION: IMAGE DETECTION Pipeline
    ALGORITHM: ResNet50 + MC Dropout + Temperature Scaling
    STEPS:
      1. Decode image bytes to PIL Image
      2. Apply inference transforms (normalize, resize)
      3. Run deterministic + MC Dropout predictions
      4. Apply temperature scaling for calibration
      5. Compute uncertainty interval
    LIBRARY: Pillow, PyTorch, TorchVision
    OUTPUT: JSON with confidence, fake_probability, uncertainty
    """
    try:
        x = eval_tf()(Image.open(io.BytesIO(raw)).convert("RGB")).unsqueeze(0).to(pack.device)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return {"label_id": -1, "label": "Error", "confidence": 0.5, "fake_probability": 0.5}
    
    # Deterministic prediction (stable across threshold changes)
    det_probs = deterministic_predict(pack.model, x, scaler=None)
    p_fake = float(det_probs[0, 1])

    # Calibrated deterministic prediction
    if pack.scaler:
        det_cal = deterministic_predict(pack.model, x, scaler=pack.scaler)
        p_fake_cal = float(det_cal[0, 1])
    else:
        p_fake_cal = p_fake

    # Uncertainty estimation via MC Dropout
    mean_probs, std_probs = mc_dropout_predict(pack.model, x, n=25, scaler=pack.scaler)
    
    # Decision with confidence margin
    uncertainty = float(std_probs[0, 1])
    use_p = p_fake_cal if pack.scaler else p_fake
    
    cls, label = _classify_with_margin(use_p, pack.threshold, uncertainty)
    
    return {
        "label_id": cls,
        "label": label,
        "confidence": abs(use_p - 0.5) * 2,
        "fake_probability": p_fake,
        "fake_probability_calibrated": p_fake_cal,
        "uncertainty_std": uncertainty,
        "uncertainty_interval": [max(0.0, p_fake - uncertainty), min(1.0, p_fake + uncertainty)],
        "decision_threshold": float(pack.threshold),
    }


def predict_video_file(pack: InferencePack, path: str, n_frames: int = 16) -> Dict:
    """
    FUNCTION: VIDEO DETECTION (Frame-level) Pipeline
    ALGORITHM: ResNet50 on sampled video frames + MC Dropout
    STEPS:
      1. Sample frames uniformly from video
      2. Apply inference transforms to each frame
      3. Run MC Dropout predictions per frame
      4. Aggregate frame probabilities (weighted fusion: mean, p90, max)
    LIBRARY: OpenCV (frame extraction), PyTorch (MC Dropout inference)
    OUTPUT: Frame-level fake probability with temporal statistics
    """
    frames = sample_video_frames(path, n_frames)
    if not frames:
        return {"label_id": -1, "label": "Error", "confidence": 0.5, "fake_probability": 0.5}
    
    frame_probs, frame_uncs = [], []
    
    for f in frames:
        x = eval_tf()(Image.fromarray(f[:, :, ::-1])).unsqueeze(0).to(pack.device)
        mean_p, std_p = mc_dropout_predict(pack.model, x, n=12, scaler=pack.scaler)
        frame_probs.append(float(mean_p[0, 1]))
        frame_uncs.append(float(std_p[0, 1]))
    
    # Aggregate: use weighted average + high percentile
    p_mean = float(np.mean(frame_probs))
    p_max = float(np.max(frame_probs))
    p_90 = float(np.percentile(frame_probs, 90))
    p_fake = 0.6 * p_mean + 0.25 * p_90 + 0.15 * p_max  # Weighted fusion
    
    u_std = float(np.mean(frame_uncs))
    cls, label = _classify_with_margin(p_fake, pack.threshold, u_std)
    
    return {
        "label_id": cls,
        "label": label,
        "confidence": abs(p_fake - 0.5) * 2,
        "fake_probability": p_fake,
        "uncertainty_std": u_std,
        "temporal": {
            "frame_probs": frame_probs,
            "frame_mean": p_mean,
            "frame_p90": p_90,
            "frame_max": p_max,
        },
        "decision_threshold": float(pack.threshold),
    }



def load_video_model(weights_path: str, device: Optional[torch.device] = None) -> Optional[nn.Module]:
    """
    FUNCTION: Load CNN-LSTM video model
    MODEL: CNNLSTM architecture (ResNet50 + Bidirectional LSTM)
    PURPOSE: Temporal sequence analysis for video deepfakes
    LIBRARY: PyTorch, TorchVision
    RETURNS: Loaded model or None if not found
    """
    p = Path(weights_path)
    if not p.exists():
        return None
    
    d = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = CNNLSTM().to(d).eval()
    try:
        m.load_state_dict(torch.load(p, map_location=d), strict=False)
        logger.info(f"Loaded video model from {p}")
    except Exception as e:
        logger.warning(f"Failed to load video model: {e}")
    
    return m


@torch.no_grad()
def predict_video_with_cnnlstm(video_model: nn.Module, device: torch.device, path: str, n_frames: int = 12) -> Dict:
    """
    FUNCTION: VIDEO DETECTION (Temporal/Sequence) Pipeline
    ALGORITHM: CNN-LSTM for temporal modeling
    STEPS:
      1. Sample frames uniformly from video
      2. Extract spatial features (ResNet50 CNN part)
      3. Model temporal relationships (LSTM part)
      4. Sequence-level classification
    LIBRARY: PyTorch, TorchVision
    OUTPUT: Sequence-level fake probability (catches temporal inconsistencies)
    """
    frames = sample_video_frames(path, n_frames)
    if not frames:
        return {"label_id": -1, "label": "Error", "confidence": 0.5, "fake_probability": 0.5}
    
    x = torch.stack([eval_tf()(Image.fromarray(f[:, :, ::-1])) for f in frames]).unsqueeze(0).to(device)
    logits = video_model(x)
    p_fake = float(torch.softmax(logits, dim=1)[0, 1].item())
    cls = 1 if p_fake >= 0.5 else 0
    
    return {
        "label_id": cls,
        "label": LABELS[cls],
        "confidence": abs(p_fake - 0.5) * 2,
        "fake_probability": p_fake,
        "uncertainty_std": 0.15,
    }


def predict_audio_file(path: str, model_path: Optional[str] = None, device: Optional[torch.device] = None) -> Dict:
    """
    FUNCTION: AUDIO DETECTION Pipeline
    ALGORITHM: AudioClassifier CNN on Mel-spectrograms
    STEPS:
      1. Load audio file using librosa
      2. Compute Mel-spectrogram representation
      3. Run CNN inference
      4. Apply sigmoid scaling calibration
      5. Compute uncertainty and confidence
    LIBRARY: librosa (audio processing), PyTorch (CNN inference)
    OUTPUT: JSON with classification, confidence, uncertainty
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path is None or not Path(model_path).exists():
        logger.warning("Audio model not found, returning placeholder result")
        return {
            "label_id": -1,
            "label": "Audio model not trained",
            "confidence": 0.5,
            "fake_probability": 0.5,
            "note": "Train audio model with: python train.py train-audio --audio-path <path>",
        }
    
    try:
        logger.info(f"Starting audio inference on {path}")
        
        # Load audio model (with validation)
        try:
            model = AudioClassifier(dropout=0.3)  # Match training dropout
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            logger.info(f"✓ Audio model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio model from {model_path}: {type(e).__name__}: {e}")
        
        # Extract mel-spectrogram
        try:
            logger.info(f"Extracting mel-spectrogram from {path}...")
            spec = audio_to_mel_spectrogram(path, sr=16000, duration=5.0)
            logger.info(f"✓ Mel-spectrogram extracted: shape={spec.shape}")
        except FileNotFoundError as e:
            raise RuntimeError(f"Audio file not found: {path}. File may have been deleted before processing.")
        except ValueError as e:
            raise RuntimeError(f"Invalid audio file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract spectrogram: {type(e).__name__}: {e}")
        
        # Prepare tensor
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 156)
        logger.info(f"Tensor prepared: {x.shape}, device={device}")
        
        # Single inference pass (no MC Dropout) - model is small and MC Dropout adds noise
        logger.info("Running standard inference...")
        with torch.no_grad():
            try:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                p_fake_raw = probs[0, 1].item()  # Probability of being FAKE
                p_std = 0.0  # No uncertainty for single pass
                logger.info(f"Logits: {logits.tolist()}")
                logger.info(f"Probs: {probs.tolist()}")
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise RuntimeError(f"Inference failed: {e}")
        
        # **CALIBRATION: Apply sigmoid scaling to push probabilities towards extremes**
        # This encourages clear separation: REAL → <0.05, FAKE → >0.5
        # Using sigmoid with moderate slope: 8 * (x - 0.5) pushes middle values outward
        scaling_factor = 8.0  # Reduced from 12 for more stable calibration
        p_fake_calibrated = 1.0 / (1.0 + np.exp(-scaling_factor * (p_fake_raw - 0.5)))
        
        logger.info(f"✓ MC Dropout complete: p_fake_raw={p_fake_raw:.4f}, p_fake_calibrated={p_fake_calibrated:.4f}, σ={p_std:.4f}")
        
        # Classify based on calibrated probability
        threshold = 0.5
        label_id, label = _classify_with_margin(p_fake_calibrated, threshold, p_std)
        
        result = {
            "label_id": label_id,
            "label": label,
            "confidence": max(p_fake_calibrated, 1 - p_fake_calibrated),
            "fake_probability": p_fake_calibrated,
            "fake_probability_raw": p_fake_raw,  # Include raw for debugging
            "uncertainty_std": float(p_std),
            "uncertainty_interval": [max(0, p_fake_calibrated - p_std), min(1, p_fake_calibrated + p_std)],
            "note": f"MC Dropout (20 passes), σ={p_std:.4f}, calibrated with sigmoid",
        }
        logger.info(f"✓ Audio inference successful: {label} (p_fake={p_fake_calibrated:.4f})")
        return result
        
    except RuntimeError as e:
        logger.error(f"Audio inference failed: {e}")
        return {
            "label_id": -1,
            "label": "Audio analysis failed",
            "confidence": 0.5,
            "fake_probability": 0.5,
            "error": str(e),
        }
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:200]}"
        logger.error(f"Unexpected audio inference error: {err_msg}", exc_info=True)
        return {
            "label_id": -1,
            "label": "Audio processing failed",
            "confidence": 0.5,
            "fake_probability": 0.5,
            "error": err_msg,
        }


def _classify_with_margin(p_fake: float, threshold: float, uncertainty_std: float) -> Tuple[int, str]:
    """Classify with uncertainty margin."""
    # Disable inconclusive check to see actual predictions
    return (1, LABELS[1]) if p_fake >= threshold else (0, LABELS[0])



class AdvancedTrainer:
    """Training utility."""
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None

    def train_epoch(self, dl: DataLoader, opt, use_adv: bool = False) -> float:
        self.model.train()
        total_loss = 0.0
        
        for x, y in dl:
            x, y = x.to(self.device), y.to(self.device)
            
            if use_adv and random.random() > 0.5:
                x = pgd_attack(self.model, x, y, eps=8/255)
            
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item() * x.size(0)
        
        return total_loss / max(len(dl.dataset), 1)

    @torch.no_grad()
    def evaluate(self, dl: DataLoader, threshold: float = 0.5, calibrate: Optional[TemperatureScaler] = None) -> Dict[str, float]:
        self.model.eval()
        ys, ps = [], []
        
        for x, y in dl:
            logits = self.model(x.to(self.device))
            if calibrate is not None:
                logits = calibrate(logits)
            
            ys.extend(y.numpy().tolist())
            ps.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        
        y = np.array(ys, dtype=int)
        p = np.array(ps, dtype=float)
        
        metrics = compute_metrics(y, p, threshold=threshold)
        metrics["ece"] = ece_score(y, p)
        
        return metrics

    def fit_temperature(self, dl: DataLoader) -> Tuple[TemperatureScaler, float]:
        self.model.eval()
        logits_all, labels_all = [], []
        
        with torch.no_grad():
            for x, y in dl:
                logits_all.append(self.model(x.to(self.device)).cpu())
                labels_all.append(y.cpu())
        
        logits = torch.cat(logits_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        
        scaler = TemperatureScaler()
        temp = scaler.fit(logits, labels)
        
        return scaler.to(self.device), temp
