"""
Инструмент сбора датасета жестов — сохранение в .mp4
======================================================
Записывает видео жестов с камеры и сохраняет как .mp4 файлы.

Установка:
    pip install opencv-python numpy

Использование:
    python collect_dataset.py

Структура сохранённых данных:
    my_dataset/
    ├── MOTHER/
    │   ├── 0.mp4
    │   ├── 1.mp4
    │   └── ...
    ├── FATHER/
    │   ├── 0.mp4
    │   └── ...
    └── classes.npy
"""

import os
import numpy as np
import cv2

# КОНФИГУРАЦИЯ

DATASET_DIR       = "./my_dataset"
SAMPLES_PER_CLASS = 60        # сколько видео на каждое слово
RECORD_SECONDS    = 2.0       # длина одной записи в секундах
FPS               = 30        # кадров в секунду
CAMERA_INDEX      = 0         # индекс камеры (0 = встроенная)
FRAME_W           = 640
FRAME_H           = 480

CLASSES = [
    "accident",
    "africa",
    "all",
    "apple",
    "basketball",
    "bed",
    "before",
    "bird",
    "birthday",
    "black"
]

FRAMES_PER_VIDEO = int(RECORD_SECONDS * FPS)   # кадров на одно видео

# ЗАПИСЬ ОДНОГО ВИДЕО

def record_video(cap, class_name: str, sample_idx: int):
    """
    Ожидает нажатия SPACE, затем записывает FRAMES_PER_VIDEO кадров.
    Возвращает:
        list[np.ndarray] — список BGR кадров
        "skip"           — пропустить класс
        None             — выход
    """

    # Ожидание нажатия SPACE
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Фон шапки
        cv2.rectangle(frame, (0, 0), (w, 75), (30, 30, 30), -1)
        cv2.putText(frame,
                    f"Sign: {class_name}   [{sample_idx + 1} / {SAMPLES_PER_CLASS}]",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame,
                    "SPACE - write   S - next word   Q - exit",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

        # Прогресс класса внизу
        done    = sample_idx
        bar_w   = int(w * done / SAMPLES_PER_CLASS)
        cv2.rectangle(frame, (0, h - 18), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 18), (bar_w, h), (0, 180, 80), -1)
        cv2.putText(frame, f"{done}/{SAMPLES_PER_CLASS}",
                    (w - 80, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)

        cv2.imshow("Dataset collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return None
        if key == ord("s"):
            return "skip"
        if key == ord(" "):
            break

    # Обратный отсчёт 3..2..1
    for countdown in range(1, 0, -1):
        deadline = cv2.getTickCount() + int(cv2.getTickFrequency() * 0.8)
        while cv2.getTickCount() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 150), 6)
            cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 150), -1)
            cv2.putText(frame, f"Get ready...  {countdown}",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (255, 255, 255), 3)
            cv2.imshow("Dataset collection", frame)
            cv2.waitKey(1)

    # Запись кадров
    recorded_frames = []

    while len(recorded_frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        recorded_frames.append(frame.copy())

        h, w      = frame.shape[:2]
        progress  = len(recorded_frames) / FRAMES_PER_VIDEO
        bar_w     = int(w * progress)

        # Красная рамка — идёт запись
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 5)

        # Шапка
        cv2.rectangle(frame, (0, 0), (w, 60), (150, 0, 0), -1)
        cv2.putText(frame, f"● REC   {class_name}",
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 2)

        # Прогресс записи
        cv2.rectangle(frame, (0, h - 22), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 22), (bar_w, h), (0, 60, 220), -1)
        cv2.putText(frame,
                    f"{len(recorded_frames)}/{FRAMES_PER_VIDEO}  "
                    f"({len(recorded_frames)/FPS:.1f}s / {RECORD_SECONDS:.1f}s)",
                    (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        cv2.imshow("Dataset collection", frame)
        cv2.waitKey(1)

    # Зелёная вспышка — записано
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), (0, 200, 0), 8)
        cv2.putText(frame, "  Written!",
                    (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 0), 4)
        cv2.imshow("Dataset collection", frame)
        cv2.waitKey(400)

    return recorded_frames


# СОХРАНЕНИЕ ВИДЕО

def save_video(frames: list, path: str):
    """Сохраняет список BGR кадров как .mp4 файл."""
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()


# ГЛАВНАЯ ФУНКЦИЯ

def main():
    print("=" * 55)
    print("  Сбор датасета жестов  →  .mp4")
    print("=" * 55)
    print(f"Классов        : {len(CLASSES)}")
    print(f"Видео / слово  : {SAMPLES_PER_CLASS}")
    print(f"Длина записи   : {RECORD_SECONDS}с  ({FRAMES_PER_VIDEO} кадров @ {FPS}fps)")
    print(f"Итого видео    : {len(CLASSES) * SAMPLES_PER_CLASS}")
    print(f"Папка          : {os.path.abspath(DATASET_DIR)}")
    print("=" * 55)

    # Создаём папки
    for cls in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

    # Сохраняем список классов сразу
    np.save(os.path.join(DATASET_DIR, "classes.npy"), np.array(CLASSES))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print(f" Камера {CAMERA_INDEX} не найдена!")
        return

    print(f"\n Камера открыта: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
          f"{int(cap.get(cv2.CAP_PROP_FPS))}fps")

    try:
        for class_name in CLASSES:
            class_dir = os.path.join(DATASET_DIR, class_name)

            # Считаем уже записанные видео
            existing = len([f for f in os.listdir(class_dir)
                            if f.endswith(".mp4")])

            print(f"\n Класс: {class_name}  "
                  f"(записано: {existing}/{SAMPLES_PER_CLASS})")

            if existing >= SAMPLES_PER_CLASS:
                print(f"    Пропускаю — уже {existing} видео")
                continue

            sample_idx = existing
            while sample_idx < SAMPLES_PER_CLASS:

                result = record_video(cap, class_name, sample_idx)

                if result is None:
                    print("\n⏹️  Выход")
                    return

                if isinstance(result, str) and result == "skip":
                    print(f"   ⏭️  Пропускаю класс {class_name}")
                    break

                # Сохраняем .mp4
                save_path = os.path.join(class_dir, f"{sample_idx}.mp4")
                save_video(result, save_path)
                size_kb = os.path.getsize(save_path) // 1024
                print(f"    {class_name}/{sample_idx}.mp4  ({size_kb} KB)")
                sample_idx += 1

        print("\n Сбор датасета завершён!")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Итоговая статистика
    print("\n Итого:")
    total = 0
    for cls in CLASSES:
        class_dir = os.path.join(DATASET_DIR, cls)
        count = len([f for f in os.listdir(class_dir) if f.endswith(".mp4")])
        total += count
        done  = "" if count >= SAMPLES_PER_CLASS else "⏳"
        bar   = "*" * count + "*" * max(0, SAMPLES_PER_CLASS - count)
        print(f"   {done} {cls:<15} {bar} {count}/{SAMPLES_PER_CLASS}")
    print(f"\n   Всего видео: {total}")


if __name__ == "__main__":
    main()