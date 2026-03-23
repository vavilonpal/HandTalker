"""
Тестирование обученной WLASL LSTM модели
=========================================
Два режима:
    1. test_on_video()   — предсказание на видеофайле
    2. test_realtime()   — предсказание с камеры в реальном времени

Запуск:
    python test_model.py
"""

import os
import sys
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Layer

# ─── КОНФИГУРАЦИЯ ────────────────────────────────────────────────────────────

MODEL_PATH = "./wlasl_archive/wlasl_lstm_final.keras"  # путь к обученной модели
CLASSES_PATH = "./wlasl_archive/label_classes.npy"  # путь к файлу с классами
MP_MODEL_DIR = "./wlasl_archive/mp_models"  # папка с .task моделями MediaPipe

SEQUENCE_LEN = 30  # должно совпадать с тем что было при обучении
NUM_FEATURES = 258  # должно совпадать с тем что было при обучении
THRESHOLD = 0.5  # минимальная уверенность для отображения предсказания

# ─── СКАЧИВАНИЕ МОДЕЛЕЙ MEDIAPIPE ────────────────────────────────────────────

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
    )
}

os.makedirs(MP_MODEL_DIR, exist_ok=True)


class ReduceSum(Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()


def ensure_model(key: str) -> str:
    filename, url = _TASK_MODELS[key]
    path = os.path.join(MP_MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"⬇️  Скачиваю {filename} ...")
        urllib.request.urlretrieve(url, path)
    return path


# ─── ИНИЦИАЛИЗАЦИЯ ДЕТЕКТОРОВ ─────────────────────────────────────────────────

def build_detectors(running_mode):
    """Создаёт детекторы MediaPipe для нужного режима (IMAGE или LIVE_STREAM)."""
    BaseOptions = mp_tasks.BaseOptions

    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("pose")),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("hand")),
            running_mode=running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return pose_det, hand_det


# ─── ИЗВЛЕЧЕНИЕ КЕЙПОИНТОВ ───────────────────────────────────────────────────

def frame_to_keypoints(frame_rgb, pose_det, hand_det) -> np.ndarray:
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Pose 33×4 = 132
    pose_res = pose_det.detect(mp_img)
    if pose_res.pose_landmarks:
        pose_vec = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]],
            dtype=np.float32,
        ).flatten()
    else:
        pose_vec = np.zeros(33 * 4, dtype=np.float32)

    # Hands left 21×3=63 + right 21×3=63
    hand_res = hand_det.detect(mp_img)
    lh_vec = np.zeros(21 * 3, dtype=np.float32)
    rh_vec = np.zeros(21 * 3, dtype=np.float32)
    for i, handedness_list in enumerate(hand_res.handedness):
        label = handedness_list[0].category_name
        vec = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]],
            dtype=np.float32,
        ).flatten()
        if label == "Left":
            lh_vec = vec
        else:
            rh_vec = vec

    result = np.concatenate([pose_vec, lh_vec, rh_vec])

    # Гарантируем ровно NUM_FEATURES
    if len(result) > NUM_FEATURES:
        result = result[:NUM_FEATURES]
    elif len(result) < NUM_FEATURES:
        result = np.pad(result, (0, NUM_FEATURES - len(result)))

    return result.astype(np.float32)


# ─── РЕЖИМ 1: ТЕСТ НА ВИДЕОФАЙЛЕ ─────────────────────────────────────────────

def test_on_video(video_path: str):
    """
    Загружает видеофайл, извлекает кейпоинты и предсказывает жест.
    Показывает Top-5 результатов.
    """
    print(f"\n📹 Тестирование на видео: {video_path}")

    if not os.path.exists(video_path):
        print(f"❌ Файл не найден: {video_path}")
        return

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ReduceSum": ReduceSum}
    )
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    pose_det, hand_det= build_detectors(VisionTaskRunningMode.IMAGE)

    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            print("❌ Видео пустое или повреждено")
            return

        # Нормализуем до SEQUENCE_LEN кадров
        total = len(frames)
        if total >= SEQUENCE_LEN:
            indices = np.linspace(0, total - 1, SEQUENCE_LEN, dtype=int)
            selected = [frames[i] for i in indices]
        else:
            selected = frames

        # Извлекаем кейпоинты
        keypoints = []
        for frame_rgb in selected:
            kp = frame_to_keypoints(frame_rgb, pose_det, hand_det)
            keypoints.append(kp)

        seq = np.array(keypoints, dtype=np.float32)

        # Паддинг если нужно
        if len(seq) < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - len(seq), NUM_FEATURES), dtype=np.float32)
            seq = np.vstack([seq, pad])

        # Предсказание
        seq = seq[np.newaxis, ...]  # (1, 30, NUM_FEATURES)
        probs = model.predict(seq, verbose=0)[0]
        top5 = probs.argsort()[::-1][:5]

        print("\n┌─────────────────────────────────────┐")
        print("│          Top-5 предсказания          │")
        print("├──────┬──────────────────┬────────────┤")
        print("│ Rank │ Жест             │ Уверенность│")
        print("├──────┼──────────────────┼────────────┤")
        for rank, idx in enumerate(top5, 1):
            bar = "█" * int(probs[idx] * 20)
            print(f"│  {rank}   │ {classes[idx]:<16} │ {probs[idx] * 100:>6.1f}%   │")
        print("└──────┴──────────────────┴────────────┘")

    finally:
        pose_det.close()
        hand_det.close()

# ─── РЕЖИМ 2: РЕАЛЬНОЕ ВРЕМЯ С КАМЕРЫ ────────────────────────────────────────

def test_realtime(camera_index: int = 0):
    """
    Запускает предсказание жестов в реальном времени с камеры.

    Управление:
        Q — выход
        C — очистить буфер последовательности
    """
    print("\n📷 Запуск реального времени (нажмите Q для выхода)")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ReduceSum": ReduceSum}
    )
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    pose_det, hand_det= build_detectors(VisionTaskRunningMode.IMAGE)

    # Скользящее окно кейпоинтов — накапливаем SEQUENCE_LEN кадров
    sequence = deque(maxlen=SEQUENCE_LEN)
    prediction = ""
    confidence = 0.0

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Извлекаем кейпоинты текущего кадра
            try:
                kp = frame_to_keypoints(frame_rgb, pose_det, hand_det)
                sequence.append(kp)
            except Exception:
                pass

            # Делаем предсказание когда накопили достаточно кадров
            if len(sequence) == SEQUENCE_LEN:
                seq = np.array(sequence, dtype=np.float32)[np.newaxis, ...]
                probs = model.predict(seq, verbose=0)[0]
                idx = probs.argmax()

                if probs[idx] >= THRESHOLD:
                    prediction = classes[idx]
                    confidence = probs[idx]
                else:
                    prediction = "..."
                    confidence = 0.0

            # ── Отрисовка UI ─────────────────────────────────────────────────
            h, w = frame.shape[:2]

            # Прогресс-бар накопления кадров
            filled = len(sequence)
            bar_w = int(w * (filled / SEQUENCE_LEN))
            cv2.rectangle(frame, (0, h - 20), (w, h), (50, 50, 50), -1)
            cv2.rectangle(frame, (0, h - 20), (bar_w, h), (0, 200, 100), -1)
            cv2.putText(frame, f"Buffer: {filled}/{SEQUENCE_LEN}",
                        (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Основное предсказание
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(frame, prediction,
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (0, 255, 100) if confidence >= THRESHOLD else (100, 100, 100), 3)

            # Уверенность
            if confidence > 0:
                conf_text = f"{confidence * 100:.1f}%"
                cv2.putText(frame, conf_text,
                            (w - 120, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 2)

            # Подсказка
            cv2.putText(frame, "Q - quit  |  C - clear buffer",
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("WLASL Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                sequence.clear()
                prediction = ""
                confidence = 0.0
                print("🔄 Буфер очищен")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_det.close()
        hand_det.close()
        print("✅ Завершено")


# ─── ТОЧКА ВХОДА ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  WLASL Model Tester")
    print("=" * 50)
    print("1 — Тест на видеофайле из датасета")
    print("2 — Реальное время с камеры")
    choice = input("\nВыберите режим (1 или 2): ").strip()

    if choice == "1":
        path = input("Путь к видеофайлу (или Enter для случайного из датасета): ").strip()
        if not path:
            # Берём случайное видео из датасета
            import glob

            videos = glob.glob("../wlasl-archive/videos/*.mp4")
            if videos:
                import random

                path = random.choice(videos)
                print(f"Случайное видео: {path}")
            else:
                print("❌ Видео не найдены в ../wlasl-archive/videos/")
                sys.exit(1)
        test_on_video(path)

    elif choice == "2":
        test_realtime(camera_index=0)

    else:
        print("❌ Неверный выбор")
