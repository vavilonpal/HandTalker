"""
WLASL (Word-Level American Sign Language) — PyTorch LSTM Training Pipeline
===========================================================================
Dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

Установка зависимостей:
    pip install torch torchvision mediapipe opencv-python scikit-learn numpy tqdm

Для GPU (CUDA):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Структура датасета:
    wlasl_archive/
    ├── WLASL_v0.3.json
    └── videos/

Кейпоинты (258 значений на кадр):
    Pose:        33 × 4 (x, y, z, visibility) = 132
    Left hand:   21 × 3 (x, y, z)             =  63
    Right hand:  21 × 3 (x, y, z)             =  63
    Total:                                     = 258
"""

import os, json, warnings, urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

DATASET_ROOT = "./wlasl_archive"
JSON_PATH = os.path.join(DATASET_ROOT, "WLASL_v0.3.json")
VIDEOS_DIR = os.path.join(DATASET_ROOT, "videos")
CACHE_DIR = "./keypoints_cache"
MODEL_DIR = "./mp_models"
CUSTOM_DATASET_DIR = "./my_dataset"

SUBSET = 10
SEQUENCE_LEN = 30
NUM_FEATURES = 258

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
PATIENCE = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in (CACHE_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

_TASK_MODELS = {
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}


def ensure_model(key):
    filename, url = _TASK_MODELS[key]
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"Скачиваю {filename} ...")
        urllib.request.urlretrieve(url, path)
        print(f"   Сохранён: {path}")
    return path


def build_detectors():
    BaseOptions = mp_tasks.BaseOptions
    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("pose")),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("hand")),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return pose_det, hand_det


def frame_to_keypoints(frame_rgb, pose_det, hand_det):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    pose_res = pose_det.detect(mp_img)
    if pose_res.pose_landmarks:
        pose_vec = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]],
            dtype=np.float32).flatten()
    else:
        pose_vec = np.zeros(33 * 4, dtype=np.float32)

    hand_res = hand_det.detect(mp_img)
    lh_vec = np.zeros(21 * 3, dtype=np.float32)
    rh_vec = np.zeros(21 * 3, dtype=np.float32)
    for i, hl in enumerate(hand_res.handedness):
        vec = np.array([[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]],
                       dtype=np.float32).flatten()
        if hl[0].category_name == "Left":
            lh_vec = vec
        else:
            rh_vec = vec

    result = np.concatenate([pose_vec, lh_vec, rh_vec])
    if len(result) > NUM_FEATURES:
        result = result[:NUM_FEATURES]
    elif len(result) < NUM_FEATURES:
        result = np.pad(result, (0, NUM_FEATURES - len(result)))
    return result.astype(np.float32)


def video_to_sequence(video_path, pose_det, hand_det, seq_len=SEQUENCE_LEN, cache_key=None):
    if cache_key is None:
        cache_key = os.path.splitext(os.path.basename(video_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        seq = np.zeros((seq_len, NUM_FEATURES), dtype=np.float32)
        np.save(cache_file, seq)
        return seq

    total = len(frames)
    if total >= seq_len:
        indices = np.linspace(0, total - 1, seq_len, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames

    kps = [frame_to_keypoints(f, pose_det, hand_det) for f in selected]
    seq = np.array(kps, dtype=np.float32)
    if len(seq) < seq_len:
        seq = np.vstack([seq, np.zeros((seq_len - len(seq), NUM_FEATURES), dtype=np.float32)])

    np.save(cache_file, seq)
    return seq


def load_wlasl_samples(json_path, videos_dir, subset):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:subset]
    glosses = [e["gloss"] for e in data]
    samples = []
    for entry in data:
        for inst in entry["instances"]:
            vp = os.path.join(videos_dir, f"{inst['video_id']}.mp4")
            if os.path.exists(vp):
                samples.append((vp, entry["gloss"]))
    print(f"Найдено {len(samples)} видео для {subset} классов")
    return samples, glosses


def load_custom_samples(dataset_dir):
    """Загружает все mp4-видео из my_dataset/<word>/*.mp4."""
    samples, glosses = [], []
    if not os.path.isdir(dataset_dir):
        print(f"Папка кастомного датасета не найдена: {dataset_dir}")
        return samples, glosses
    for word in sorted(os.listdir(dataset_dir)):
        word_dir = os.path.join(dataset_dir, word)
        if not os.path.isdir(word_dir):
            continue
        glosses.append(word)
        for fname in os.listdir(word_dir):
            if fname.lower().endswith(".mp4"):
                samples.append((os.path.join(word_dir, fname), word))
    print(f"Кастомный датасет: {len(samples)} видео для {len(glosses)} классов")
    return samples, glosses


def augment_sequence(seq):
    aug = seq.copy()
    if np.random.rand() < 0.5:
        aug += np.random.normal(0, 0.005, aug.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        aug *= np.random.uniform(0.9, 1.1)
    if np.random.rand() < 0.5:
        aug = np.roll(aug, np.random.randint(-3, 3), axis=0)
    if np.random.rand() < 0.3:
        aug[:, 0::4] = 1.0 - aug[:, 0::4]
    return np.clip(aug, -1.0, 2.0).astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = augment_sequence(x)
        return torch.tensor(x, dtype=torch.float32), self.y[idx]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)  # (B, seq, 1)
        return (x * weights).sum(dim=1)  # (B, hidden)


class WLASLModel(nn.Module):
    def __init__(self, num_classes, input_size=NUM_FEATURES,
                 hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out = hidden_size * 2
        self.drop1 = nn.Dropout(dropout)
        self.attn = AttentionLayer(lstm_out)
        self.fc1 = nn.Linear(lstm_out, 128)
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop1(out)
        ctx = self.attn(out)
        out = F.relu(self.fc1(ctx))
        out = self.drop2(out)
        return self.fc2(out)


# Локальные детекторы для каждого потока
_thread_local = threading.local()

def get_thread_detectors():
    """Каждый поток получает свои детекторы."""
    if not hasattr(_thread_local, "detectors"):
        _thread_local.detectors = build_detectors()
    return _thread_local.detectors


def process_one(args):
    """Обрабатывает одно видео в отдельном потоке."""
    video_path, gloss = args
    try:
        # Collision-free cache key: custom videos get prefix "custom_<word>_<stem>"
        stem = os.path.splitext(os.path.basename(video_path))[0]
        parent = os.path.basename(os.path.dirname(video_path))
        norm = str(video_path).replace("\\", "/")
        if "/my_dataset/" in norm or norm.endswith("/my_dataset"):
            cache_key = f"custom_{parent}_{stem}"
        else:
            cache_key = stem
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.npy")

        if os.path.exists(cache_file):
            seq = np.load(cache_file)
        else:
            pose_det, hand_det = get_thread_detectors()
            seq = video_to_sequence(video_path, pose_det, hand_det, cache_key=cache_key)

        if seq.shape != (SEQUENCE_LEN, NUM_FEATURES):
            return None, None
        if np.mean(seq) < 1e-6:
            return None, None

        return seq, gloss
    except Exception:
        return None, None


def build_dataset(samples: list, glosses: list):
    le = LabelEncoder()
    le.fit(glosses)

    cached = sum(
        1 for video_path, _ in samples
        if os.path.exists(os.path.join(CACHE_DIR,
            os.path.splitext(os.path.basename(video_path))[0] + ".npy"))
    )
    need_extraction = cached < len(samples)

    if need_extraction:
        print(f"Кэш: {cached}/{len(samples)} — извлекаю кейпоинты ...")
    else:
        print(f"Все {cached} кейпоинтов в кэше — загружаю ...")

    X, y_labels, failed = [], [], 0

    # 4 потока — оптимально для MediaPipe на CPU
    NUM_WORKERS = 4

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one, s): s for s in samples}
        for future in tqdm(as_completed(futures),
                           total=len(samples),
                           desc="Загрузка кейпоинтов"):
            seq, gloss = future.result()
            if seq is None:
                failed += 1
            else:
                X.append(seq)
                y_labels.append(gloss)

    if failed:
        print(f"Пропущено {failed} видео")

    X     = np.array(X, dtype=np.float32)
    y_enc = le.transform(y_labels)

    print(f"Датасет готов: X={X.shape}  классов={len(glosses)}")
    return X, y_enc, le

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = correct5 = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        top5 = logits.topk(min(5, logits.size(1)), dim=1).indices
        correct5 += sum(yb[i].item() in top5[i].tolist() for i in range(len(yb)))
        total += len(yb)
    return total_loss / total, correct / total, correct5 / total


def main():
    print("=" * 60)
    print("  WLASL — PyTorch Training Pipeline")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    wlasl_samples, wlasl_glosses = load_wlasl_samples(JSON_PATH, VIDEOS_DIR, SUBSET)
    custom_samples, custom_glosses = load_custom_samples(CUSTOM_DATASET_DIR)

    # Merge: add custom glosses not already in WLASL to avoid duplicates
    extra_glosses = [g for g in custom_glosses if g not in wlasl_glosses]
    all_glosses = wlasl_glosses + extra_glosses
    all_samples = wlasl_samples + custom_samples
    print(f"Всего после объединения: {len(all_samples)} видео, {len(all_glosses)} классов")

    X, y_enc, le = build_dataset(all_samples, all_glosses)
    np.save("label_classes.npy", le.classes_)
    print("label_classes.npy сохранён")

    num_classes = len(all_glosses)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y_enc, test_size=TEST_SPLIT, stratify=y_enc, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=VALIDATION_SPLIT / (1.0 - TEST_SPLIT),
        stratify=y_tmp, random_state=42)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(SignDataset(X_train, y_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(SignDataset(X_val, y_val, augment=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(SignDataset(X_test, y_test, augment=False),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WLASLModel(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=7, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_acc = 0.0
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_top5 = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train {tr_loss:.4f}/{tr_acc * 100:.1f}% | "
              f"val {vl_loss:.4f}/{vl_acc * 100:.1f}% top5={vl_top5 * 100:.1f}%", end="")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": vl_acc, "num_classes": num_classes}, "best_wlasl_lstm.pt")
            print("  сохранено")
            patience_cnt = 0
        else:
            print()
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping на эпохе {epoch}")
                break

    # Финальный тест
    ckpt = torch.load("best_wlasl_lstm.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    ts_loss, ts_acc, ts_top5 = eval_epoch(model, test_loader, criterion)
    print("\nТест:")
    print(f"   Loss      : {ts_loss:.4f}")
    print(f"   Top-1 Acc : {ts_acc * 100:.1f}%")
    print(f"   Top-5 Acc : {ts_top5 * 100:.1f}%")

    torch.save({"model_state": model.state_dict(), "num_classes": num_classes,
                "input_size": NUM_FEATURES, "seq_len": SEQUENCE_LEN}, "wlasl_lstm_final.pt")
    print("\nМодель сохранена: wlasl_lstm_final.pt")


def predict_video(video_path, model_path="best_wlasl_lstm.pt"):
    classes = np.load("label_classes.npy", allow_pickle=True)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = WLASLModel(num_classes=ckpt["num_classes"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pose_det, hand_det = build_detectors()
    try:
        seq = video_to_sequence(video_path, pose_det, hand_det)
    finally:
        pose_det.close()
        hand_det.close()

    x = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    top5 = probs.argsort()[::-1][:5]
    print(f"\nВидео: {video_path}\nTop-5:")
    for i, idx in enumerate(top5, 1):
        print(f"  {i}. {classes[idx]:<25} {probs[idx] * 100:.1f}%")
    return classes[top5[0]]


if __name__ == "__main__":
    main()
    # predict_video("../wlasl_archive/videos/00001.mp4")
