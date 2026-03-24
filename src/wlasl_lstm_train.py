"""
WLASL (Word-Level American Sign Language) — LSTM Training Pipeline
====================================================================
Dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

Установка зависимостей:
    pip install tensorflow mediapipe opencv-python scikit-learn numpy tqdm

Структура датасета после скачивания:
    wlasl-processed/
    ├── WLASL_v0.3.json       ← метаданные (gloss → video ids)
    └── videos/               ← все .mp4 файлы (video_id.mp4)

Кейпоинты MediaPipe Tasks API (1662 значений на кадр):
    Pose:        33 landmarks × 4 (x, y, z, visibility) = 132
    Face:       468 landmarks × 3 (x, y, z)             = 1404
    Left hand:   21 landmarks × 3 (x, y, z)             =  63
    Right hand:  21 landmarks × 3 (x, y, z)             =  63
    ─────────────────────────────────────────────────────────
    Итого:                                               = 1662
"""

# ─── ИМПОРТЫ ─────────────────────────────────────────────────────────────────

"""
WLASL (Word-Level American Sign Language) — LSTM Training Pipeline
====================================================================
Dataset: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

Установка зависимостей:
    pip install tensorflow mediapipe opencv-python scikit-learn numpy tqdm

Структура датасета после скачивания:
    wlasl-processed/
    ├── WLASL_v0.3.json       ← метаданные (gloss → video ids)
    └── videos/               ← все .mp4 файлы (video_id.mp4)

Кейпоинты MediaPipe Tasks API (1662 значений на кадр):
    Pose:        33 landmarks × 4 (x, y, z, visibility) = 132
    Face:       468 landmarks × 3 (x, y, z)             = 1404
    Left hand:   21 landmarks × 3 (x, y, z)             =  63
    Right hand:  21 landmarks × 3 (x, y, z)             =  63
    ─────────────────────────────────────────────────────────
    Итого:                                               = 1662
"""

# ─── ИМПОРТЫ ─────────────────────────────────────────────────────────────────

import os
import json
import warnings
import urllib.request
import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Softmax, Lambda, Multiply, Layer
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─── КОНФИГУРАЦИЯ ────────────────────────────────────────────────────────────

DATASET_ROOT = "./wlasl_archive"
JSON_PATH = os.path.join(DATASET_ROOT, "WLASL_v0.3.json")
VIDEOS_DIR = os.path.join(DATASET_ROOT, "videos")
CACHE_DIR = "./keypoints_cache"  # .npy кэш кейпоинтов
MODEL_DIR = "./mp_models"  # .task файлы MediaPipe

SUBSET = 40  # 100 / 300 / 1000 / 2000 классов
SEQUENCE_LEN = 30  # кадров на один пример

# убираем полностью лицо и у нас остается 132 + 63 + 63 = 258 features
NUM_FEATURES = 258  # размер вектора кейпоинтов

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

for d in (CACHE_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ─── 1. СКАЧИВАНИЕ .task МОДЕЛЕЙ ─────────────────────────────────────────────

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
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
    ),
}


def ensure_model(key: str) -> str:
    """Возвращает путь к .task файлу, скачивая его при необходимости."""
    filename, url = _TASK_MODELS[key]
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"⬇️  Скачиваю {filename} ...")
        urllib.request.urlretrieve(url, path)
        print(f"   ✅ {filename} сохранён → {path}")
    return path


# ─── 2. ПОСТРОЕНИЕ ДЕТЕКТОРОВ MEDIAPIPE TASKS API ────────────────────────────

def build_detectors():
    """
    Создаёт три детектора Tasks API в режиме IMAGE:
        pose_det  — PoseLandmarker
        hand_det  — HandLandmarker (до 2 рук)
    Вызывайте .close() после завершения работы.
    """
    BaseOptions = mp_tasks.BaseOptions

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("pose")),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model("hand")),
        running_mode=VisionTaskRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)

    return pose_det, hand_det


# ─── 3. ИЗВЛЕЧЕНИЕ ВЕКТОРА КЕЙПОИНТОВ ───────────────────────────────────────

def frame_to_keypoints(
        frame_rgb: np.ndarray,
        pose_det,
        hand_det,
) -> np.ndarray:
    """
    Принимает один RGB кадр (H×W×3 uint8).
    Возвращает вектор длиной 1692.
    """
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # ── Pose: 33 × 4 = 132 ───────────────────────────────────────────────────
    pose_res = pose_det.detect(mp_img)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        pose_vec = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
            dtype=np.float32,
        ).flatten()
    else:
        pose_vec = np.zeros(33 * 4, dtype=np.float32)

    # ── Hands: left 21×3=63 + right 21×3=63 ──────────────────────────────────
    hand_res = hand_det.detect(mp_img)
    lh_vec = np.zeros(21 * 3, dtype=np.float32)
    rh_vec = np.zeros(21 * 3, dtype=np.float32)

    for i, handedness_list in enumerate(hand_res.handedness):
        label = handedness_list[0].category_name  # "Left" или "Right"
        lms = hand_res.hand_landmarks[i]
        vec = np.array(
            [[lm.x, lm.y, lm.z] for lm in lms],
            dtype=np.float32,
        ).flatten()
        if label == "Left":
            lh_vec = vec
        else:
            rh_vec = vec

    # ── Конкатенация: 132 + 63 + 63 = 258 features
    return np.concatenate([pose_vec, lh_vec, rh_vec])


# ─── 4. ВИДЕО → ПОСЛЕДОВАТЕЛЬНОСТЬ КЕЙПОИНТОВ ────────────────────────────────

def video_to_sequence(
        video_path: str,
        pose_det,
        hand_det,
        seq_len: int = SEQUENCE_LEN,
) -> np.ndarray:
    """
    Читает видео, извлекает кейпоинты через Tasks API.
    Возвращает массив (seq_len, 1692).

    Стратегия нормализации длины:
      кадров >= seq_len → равномерная выборка seq_len кадров
      кадров  < seq_len → паддинг нулями в конце
    """
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

    if len(frames) == 0:
        seq = np.zeros((seq_len, NUM_FEATURES), dtype=np.float32)
        np.save(cache_file, seq)
        return seq

    total = len(frames)
    if total >= seq_len:
        indices = np.linspace(0, total - 1, seq_len, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames

    keypoints = []
    for frame_rgb in selected:
        kp = frame_to_keypoints(frame_rgb, pose_det, hand_det)
        keypoints.append(kp)

    seq = np.array(keypoints, dtype=np.float32)

    if len(seq) < seq_len:
        pad = np.zeros((seq_len - len(seq), NUM_FEATURES), dtype=np.float32)
        seq = np.vstack([seq, pad])

    np.save(cache_file, seq)
    return seq


# ─── 5. ЗАГРУЗКА МЕТАДАННЫХ WLASL ────────────────────────────────────────────

def load_wlasl_samples(json_path: str, videos_dir: str, subset: int):
    """
    Читает WLASL_v0.3.json.
    Возвращает:
        samples — список (video_path, gloss)
        glosses — список уникальных меток (первые `subset`)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:subset]
    glosses = [entry["gloss"] for entry in data]
    samples = []

    for entry in data:
        gloss = entry["gloss"]
        for instance in entry["instances"]:
            video_path = os.path.join(videos_dir, f"{instance['video_id']}.mp4")
            if os.path.exists(video_path):
                samples.append((video_path, gloss))

    print(f"✅ Найдено {len(samples)} видео для {subset} классов")
    return samples, glosses


# ─── 6. ПОСТРОЕНИЕ ДАТАСЕТА ───────────────────────────────────────────────────

def build_dataset(samples: list, glosses: list):
    le = LabelEncoder()
    le.fit(glosses)

    X, y_labels = [], []
    failed = 0

    # Проверяем сколько кэшированных файлов уже есть
    cached = sum(
        1 for video_path, _ in samples
        if os.path.exists(os.path.join(CACHE_DIR,
                                       os.path.splitext(os.path.basename(video_path))[0] + ".npy"))
    )
    need_extraction = cached < len(samples)

    if need_extraction:
        print(f"📦 Кэш: {cached}/{len(samples)} — запускаю извлечение кейпоинтов ...")
        print("🔧 Инициализация MediaPipe Tasks API детекторов ...")
        pose_det, hand_det = build_detectors()
    else:
        print(f"✅ Все {cached} кейпоинтов уже в кэше — пропускаю MediaPipe, сразу загружаю ...")
        pose_det = hand_det = None

    try:
        for video_path, gloss in tqdm(samples, desc="Загрузка кейпоинтов"):
            try:
                cache_file = os.path.join(CACHE_DIR,
                                          os.path.splitext(os.path.basename(video_path))[0] + ".npy")

                if os.path.exists(cache_file):
                    # Берём из кэша без MediaPipe
                    seq = np.load(cache_file)
                else:
                    # Извлекаем через MediaPipe
                    seq = video_to_sequence(video_path, pose_det, hand_det)

                # проверка на нули, если в видео ничего не задетектило то его просто пропускаем
                if np.mean(seq) < 1e-6:
                    continue  # пропускаем мусорное видео

                X.append(seq)
                y_labels.append(gloss)
            except Exception:
                failed += 1
    finally:
        if need_extraction and pose_det is not None:
            pose_det.close()
            hand_det.close()

    if failed:
        print(f"⚠️  Пропущено {failed} видео из-за ошибок")

    X = np.array(X, dtype=np.float32)
    y_enc = le.transform(y_labels)
    y_cat = to_categorical(y_enc, num_classes=len(glosses))

    print(f"✅ Датасет готов: X={X.shape}  y={y_cat.shape}")
    return X, y_cat, le


# ─── АУГМЕНТАЦИЯ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ────────────────────────────────────────

def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Применяет случайные преобразования к последовательности кейпоинтов.
    Помогает модели обобщать на новых людей и условия съёмки.
    """
    aug = seq.copy()

    # 1. Случайный шум (имитирует дрожание рук)
    if np.random.rand() < 0.5:
        aug += np.random.normal(0, 0.005, aug.shape).astype(np.float32)

    # 2. Случайное масштабирование (разные расстояния до камеры)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        aug *= scale

    # 3. Случайный сдвиг по времени (разная скорость жеста)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 3)
        aug = np.roll(aug, shift, axis=0)

    # 4. Случайное зеркалирование по X (левша/правша)
    if np.random.rand() < 0.3:
        # Меняем знак у x-координат (каждая 3-я или 4-я колонка)
        aug_flip = aug.copy()
        # Pose: x-координаты на позициях 0,4,8... (каждые 4 значения)
        aug_flip[:, 0::4] = 1.0 - aug_flip[:, 0::4]
        aug = aug_flip

    return np.clip(aug, -1.0, 2.0).astype(np.float32)


def augment_dataset(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple:
    """
    Увеличивает датасет в `factor` раз через аугментацию.
    factor=3 → датасет утраивается.
    """
    X_aug, y_aug = [X], [y]
    for _ in range(factor - 1):
        X_new = np.array([augment_sequence(seq) for seq in X], dtype=np.float32)
        X_aug.append(X_new)
        y_aug.append(y)
    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)

    # Перемешиваем
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


class ReduceSum(Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()


# ─── 7. МОДЕЛЬ ────────────────────────────────────────────────────────────────
# ref: поменял relu параметр модели на tanh
def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQUENCE_LEN, NUM_FEATURES))

    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.5)(x)  # увеличен с 0.3 → 0.5

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.5)(x)  # увеличен с 0.3 → 0.5

    # Attention
    attention = Dense(1, activation="tanh")(x)
    attention = tf.keras.layers.Softmax(axis=1)(attention)

    x = Multiply()([x, attention])
    x = ReduceSum()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)  # увеличен с 0.3 → 0.5

    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5)
        ],
    )

    return model


# ─── 8. КОЛЛБЭКИ ─────────────────────────────────────────────────────────────

def get_callbacks():
    return [
        ModelCheckpoint(
            filepath="best_wlasl_lstm.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard(log_dir="./logs"),
    ]


# ─── 9. ГЛАВНАЯ ФУНКЦИЯ ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  WLASL LSTM — MediaPipe Tasks API Training Pipeline")
    print(f"  MediaPipe  : {mp.__version__}")
    print(f"  TensorFlow : {tf.__version__}")
    print("=" * 60)

    samples, glosses = load_wlasl_samples(JSON_PATH, VIDEOS_DIR, SUBSET)

    X, y, label_encoder = build_dataset(samples, glosses)
    print(f"Всего примеров: {X.shape[0]}")
    print(f"Среднее на класс: {X.shape[0] / len(glosses):.1f}")
    np.save("label_classes.npy", label_encoder.classes_)
    print("💾 Классы сохранены → label_classes.npy")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        stratify=y.argmax(axis=1),
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=VALIDATION_SPLIT / (1.0 - TEST_SPLIT),
        stratify=y_tmp.argmax(axis=1),
        random_state=42,
    )

    # Аугментация только тренировочного сета (val и test не трогаем!)
    print(f"📈 Аугментация: {X_train.shape[0]} → ", end="")
    X_train, y_train = augment_dataset(X_train, y_train, factor=3)
    print(f"{X_train.shape[0]} примеров")

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    model = build_model(num_classes=len(glosses))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
    )

    print("\n📊 Результаты на тестовой выборке:")
    test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Loss      : {test_loss:.4f}")
    print(f"   Top-1 Acc : {test_acc * 100:.1f}%")
    print(f"   Top-5 Acc : {test_top5 * 100:.1f}%")

    model.save("wlasl_lstm_final.keras")
    print("\n✅ Модель сохранена → wlasl_lstm_final.keras")

    return history


# ─── 10. ИНФЕРЕНС ────────────────────────────────────────────────────────────

def predict_video(video_path: str, model_path: str = "best_wlasl_lstm.keras"):
    """
    Предсказывает глоссу для одного видеофайла.
    Пример: predict_video("./wlasl-processed/videos/00001.mp4")
    """
    model = tf.keras.models.load_model(model_path)
    classes = np.load("label_classes.npy", allow_pickle=True)

    print("🔧 Инициализация детекторов для инференса ...")
    pose_det, hand_det = build_detectors()

    try:
        seq = video_to_sequence(video_path, pose_det, hand_det)
    finally:
        pose_det.close()
        hand_det.close()

    seq = seq[np.newaxis, ...]
    probs = model.predict(seq, verbose=0)[0]
    top5 = probs.argsort()[::-1][:5]

    print(f"\nВидео: {video_path}")
    print("Top-5 предсказания:")
    for rank, idx in enumerate(top5, 1):
        print(f"  {rank}. {classes[idx]:<25} {probs[idx] * 100:.1f}%")

    return classes[top5[0]]


# ─── ТОЧКА ВХОДА ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    history = main()

    # Пример инференса после обучения:
    # predict_video("./wlasl-processed/videos/00001.mp4")
