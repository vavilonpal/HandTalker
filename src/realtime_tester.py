"""
Real-Time Sign Language Tester (PyTorch)
=========================================
Два режима:
    1. Webcam (реальное время) — показывает предсказание текущего жеста,
       строит предложение на экране без дубликатов подряд, нажмите S чтобы
       сохранить предложение, Q чтобы выйти.

    2. Video file — анализирует видеофайл скользящим окном, возвращает
       полное предложение из распознанных жестов без дубликатов.

Запуск:
    python realtime_tester.py                  # меню
    python realtime_tester.py --video path.mp4 # конкретный файл

Требования: torch, mediapipe, opencv-python, numpy
"""

import os
import sys
import argparse
import urllib.request
from collections import deque

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── КОНФИГУРАЦИЯ ────────────────────────────────────────────────────────────

MODEL_PATH   = "./wlasl_lstm_final.pt"   # PyTorch-модель
CLASSES_PATH = "./label_classes.npy"     # массив классов
MP_MODEL_DIR = "./mp_models"             # папка .task моделей MediaPipe

SEQUENCE_LEN = 30       # совпадает с обучением
NUM_FEATURES = 258      # совпадает с обучением
THRESHOLD    = 0.85     # минимальная уверенность для принятия предсказания
PRED_EVERY   = 8        # делать предсказание раз в N кадров (снижает нагрузку)
CONFIRM_REPS = 3        # сколько раз подряд предсказание должно совпасть → слово в предложение

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MEDIAPIPE МОДЕЛИ ────────────────────────────────────────────────────────

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
os.makedirs(MP_MODEL_DIR, exist_ok=True)


def ensure_model(key: str) -> str:
    filename, url = _TASK_MODELS[key]
    path = os.path.join(MP_MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"⬇️  Скачиваю {filename} ...")
        urllib.request.urlretrieve(url, path)
        print(f"   ✅ {path}")
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


# ─── ИЗВЛЕЧЕНИЕ КЕЙПОИНТОВ ───────────────────────────────────────────────────

def frame_to_keypoints(frame_rgb: np.ndarray, pose_det, hand_det) -> np.ndarray:
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    pose_res = pose_det.detect(mp_img)
    if pose_res.pose_landmarks:
        pose_vec = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]],
            dtype=np.float32,
        ).flatten()
    else:
        pose_vec = np.zeros(33 * 4, dtype=np.float32)

    hand_res = hand_det.detect(mp_img)
    lh_vec = np.zeros(21 * 3, dtype=np.float32)
    rh_vec = np.zeros(21 * 3, dtype=np.float32)
    for i, hl in enumerate(hand_res.handedness):
        vec = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]],
            dtype=np.float32,
        ).flatten()
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


# ─── PYTORCH МОДЕЛЬ (копия архитектуры из обучения) ──────────────────────────

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class WLASLModel(nn.Module):
    def __init__(self, num_classes: int, input_size: int = NUM_FEATURES,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.5):
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
        self.attn  = AttentionLayer(lstm_out)
        self.fc1   = nn.Linear(lstm_out, 128)
        self.drop2 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.drop1(out)
        ctx    = self.attn(out)
        out    = F.relu(self.fc1(ctx))
        out    = self.drop2(out)
        return self.fc2(out)


# ─── ЗАГРУЗКА МОДЕЛИ ─────────────────────────────────────────────────────────

def load_model(model_path: str = MODEL_PATH, classes_path: str = CLASSES_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Классы не найдены: {classes_path}")

    classes = np.load(classes_path, allow_pickle=True)
    ckpt    = torch.load(model_path, map_location=DEVICE)

    model   = WLASLModel(num_classes=ckpt["num_classes"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Модель загружена | классов: {ckpt['num_classes']} | device: {DEVICE}")
    return model, classes


# ─── ОБЩАЯ УТИЛИТА: ИНФЕРЕНС ──────────────────────────────────────────────────

@torch.no_grad()
def predict_sequence(seq: np.ndarray, model, classes) -> tuple[str, float]:
    """seq: (SEQUENCE_LEN, NUM_FEATURES) → (word, confidence)"""
    x     = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(DEVICE)
    probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    idx   = int(probs.argmax())
    return str(classes[idx]), float(probs[idx])


# ─── ВСПОМОГАТЕЛЬНАЯ: ДОБАВИТЬ СЛОВО В ПРЕДЛОЖЕНИЕ (без дублей подряд) ───────

def append_word(sentence: list, word: str) -> list:
    """Добавляет слово если оно отличается от последнего в предложении."""
    if not sentence or sentence[-1] != word:
        sentence.append(word)
    return sentence


# ─── РЕЖИМ 1: РЕАЛЬНОЕ ВРЕМЯ С КАМЕРЫ ───────────────────────────────────────

def run_realtime(camera_index: int = 0):
    """
    Предсказание жестов с камеры в реальном времени.

    Управление:
        Q — выход и вывод итогового предложения
        C — очистить буфер и предложение
        S — сохранить текущее предложение в файл sentence.txt
    """
    print("\n📷 Реальное время (Q — выход, C — очистить, S — сохранить)")

    model, classes = load_model()
    pose_det, hand_det = build_detectors()

    sequence     = deque(maxlen=SEQUENCE_LEN)
    sentence     = []
    prediction   = ""
    confidence   = 0.0
    confirm_buf  = deque(maxlen=CONFIRM_REPS)
    frame_count  = 0

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                kp = frame_to_keypoints(frame_rgb, pose_det, hand_det)
                sequence.append(kp)
            except Exception:
                pass

            # Предсказание раз в PRED_EVERY кадров
            if len(sequence) == SEQUENCE_LEN and frame_count % PRED_EVERY == 0:
                seq_arr = np.array(sequence, dtype=np.float32)
                word, conf = predict_sequence(seq_arr, model, classes)

                if conf >= THRESHOLD:
                    prediction = word
                    confidence = conf
                    confirm_buf.append(word)
                    if (len(confirm_buf) == CONFIRM_REPS
                            and len(set(confirm_buf)) == 1):
                        append_word(sentence, word)
                        confirm_buf.clear()
                else:
                    prediction = "..."
                    confidence = 0.0
                    confirm_buf.clear()

            # ── Отрисовка ────────────────────────────────────────────────────
            h, w = frame.shape[:2]

            # Верхняя панель — текущее предсказание
            cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 15), -1)
            color = (0, 230, 100) if confidence >= THRESHOLD else (120, 120, 120)
            cv2.putText(frame, prediction,
                        (20, 62), cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 3)
            if confidence > 0:
                cv2.putText(frame, f"{confidence * 100:.1f}%",
                            (w - 130, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255, 200, 0), 2)

            # Предложение (снизу)
            sentence_str = " ".join(sentence) if sentence else "(пусто)"
            cv2.rectangle(frame, (0, h - 95), (w, h - 50), (25, 25, 25), -1)
            cv2.putText(frame, "Sentence: " + sentence_str,
                        (10, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (200, 240, 255), 2)

            # Прогресс-бар буфера
            filled = len(sequence)
            bar_w  = int(w * (filled / SEQUENCE_LEN))
            cv2.rectangle(frame, (0, h - 50), (w, h - 28), (40, 40, 40), -1)
            cv2.rectangle(frame, (0, h - 50), (bar_w, h - 28), (0, 170, 90), -1)
            cv2.putText(frame, f"Buffer {filled}/{SEQUENCE_LEN}",
                        (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1)

            # Подсказка
            cv2.rectangle(frame, (0, h - 28), (w, h), (10, 10, 10), -1)
            cv2.putText(frame, "Q - quit  |  C - clear  |  S - save",
                        (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (160, 160, 160), 1)

            cv2.imshow("Sign Language — Real-Time", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                sequence.clear()
                sentence.clear()
                confirm_buf.clear()
                prediction = ""
                confidence = 0.0
                print("🔄 Буфер и предложение очищены")
            elif key == ord("s"):
                with open("sentence.txt", "w", encoding="utf-8") as f:
                    f.write(" ".join(sentence))
                print(f"💾 Сохранено → sentence.txt: {' '.join(sentence)}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_det.close()
        hand_det.close()

    result = " ".join(sentence)
    print(f"\n📝 Итоговое предложение: {result if result else '(ничего не распознано)'}")
    return result


# ─── РЕЖИМ 2: АНАЛИЗ ВИДЕОФАЙЛА → ПРЕДЛОЖЕНИЕ ────────────────────────────────

def analyze_video(video_path: str, show_window: bool = True) -> str:
    """
    Анализирует видеофайл скользящим окном SEQUENCE_LEN кадров.
    Строит предложение из слов без дубликатов подряд.
    Возвращает строку — предложение.
    """
    print(f"\n📹 Анализ видео: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Файл не найден: {video_path}")

    model, classes = load_model()
    pose_det, hand_det = build_detectors()

    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"   Кадров: {total_frames} | FPS: {fps:.1f}")

    sequence    = deque(maxlen=SEQUENCE_LEN)
    sentence    = []
    confirm_buf = deque(maxlen=CONFIRM_REPS)
    frame_count = 0
    last_word   = ""
    last_conf   = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                kp = frame_to_keypoints(frame_rgb, pose_det, hand_det)
                sequence.append(kp)
            except Exception:
                pass

            if len(sequence) == SEQUENCE_LEN and frame_count % PRED_EVERY == 0:
                seq_arr = np.array(sequence, dtype=np.float32)
                word, conf = predict_sequence(seq_arr, model, classes)

                if conf >= THRESHOLD:
                    last_word = word
                    last_conf = conf
                    confirm_buf.append(word)
                    if len(confirm_buf) == CONFIRM_REPS and len(set(confirm_buf)) == 1:
                        append_word(sentence, word)
                        confirm_buf.clear()
                else:
                    last_word = "..."
                    last_conf = 0.0
                    confirm_buf.clear()

            if show_window:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 85), (15, 15, 15), -1)
                color = (0, 230, 100) if last_conf >= THRESHOLD else (120, 120, 120)
                cv2.putText(frame, last_word,
                            (20, 58), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
                if last_conf > 0:
                    cv2.putText(frame, f"{last_conf * 100:.1f}%",
                                (w - 120, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (255, 200, 0), 2)

                sentence_str = " ".join(sentence) if sentence else ""
                cv2.rectangle(frame, (0, h - 55), (w, h - 25), (25, 25, 25), -1)
                cv2.putText(frame, "Sentence: " + sentence_str,
                            (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 240, 255), 2)

                progress   = frame_count / max(total_frames, 1)
                prog_bar_w = int(w * progress)
                cv2.rectangle(frame, (0, h - 25), (w, h), (30, 30, 30), -1)
                cv2.rectangle(frame, (0, h - 25), (prog_bar_w, h), (80, 60, 200), -1)
                cv2.putText(frame, f"{frame_count}/{total_frames}",
                            (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)

                cv2.imshow("Sign Language — Video Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("⏹️  Прерван пользователем")
                    break

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        pose_det.close()
        hand_det.close()

    result = " ".join(sentence)
    print(f"\n📝 Распознанное предложение: {result if result else '(ничего не распознано)'}")
    return result


# ─── ТОЧКА ВХОДА ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time sign language tester (PyTorch)")
    parser.add_argument("--video",     type=str, default="",
                        help="Путь к видеофайлу (если не указан — режим камеры)")
    parser.add_argument("--camera",    type=int, default=0,
                        help="Индекс камеры (default: 0)")
    parser.add_argument("--no-window", action="store_true",
                        help="Не показывать окно при анализе видео")
    args = parser.parse_args()

    print("=" * 55)
    print("  Sign Language Real-Time Tester  (PyTorch)")
    print(f"  Модель  : {MODEL_PATH}")
    print(f"  Классы  : {CLASSES_PATH}")
    print(f"  Device  : {DEVICE}")
    print("=" * 55)

    if args.video:
        sentence = analyze_video(args.video, show_window=not args.no_window)
        print(f"\n✅ Результат: \"{sentence}\"")
    else:
        print("1 — Веб-камера (реальное время)")
        print("2 — Анализ видеофайла")
        choice = input("\nВыберите режим (1 / 2): ").strip()

        if choice == "1":
            run_realtime(camera_index=args.camera)
        elif choice == "2":
            path = input("Путь к видеофайлу: ").strip()
            sentence = analyze_video(path)
            print(f"\n✅ Результат: \"{sentence}\"")
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    main()
