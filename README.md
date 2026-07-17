# HandTalker

Система распознавания языка жестов с синтезом речи и анализом эмоций.

Принимает видео с жестами американского жестового языка (ASL), распознаёт показанные слова, определяет доминирующую эмоцию говорящего, озвучивает распознанный текст и накладывает аудио обратно на исходное видео.

---

## Содержание

- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Данные](#данные)
- [Обучение модели](#обучение-модели)
- [Тестирование в реальном времени](#тестирование-в-реальном-времени)
- [Сервисный слой](#сервисный-слой)
- [Результаты обучения](#результаты-обучения)

---

## Архитектура

Пайплайн состоит из четырёх последовательных шагов.

```
Входное видео
    |
    v
[1] SignToTextService      -- PyTorch Bi-LSTM + Attention
    Извлечение кейпоинтов MediaPipe -> распознавание жестов -> предложение
    |
    v
[2] EmotionService         -- DeepFace
    Анализ лица по кадрам -> доминирующая эмоция видео
    |
    v
[3] TTSService             -- Coqui TTS (xtts_v2)
    Текст + эмоция -> WAV-файл (темп речи подстраивается под эмоцию)
    |
    v
[4] AudioOverlayService    -- FFmpeg
    Исходное видео + WAV -> итоговый MP4
```

### Модель распознавания жестов

Входные признаки: 258 значений на кадр.

| Источник | Точки | Координаты | Значений |
|---|---|---|---|
| Поза тела | 33 | x, y, z, visibility | 132 |
| Левая рука | 21 | x, y, z | 63 |
| Правая рука | 21 | x, y, z | 63 |
| Итого | | | 258 |

- Длина последовательности: 30 кадров на жест
- Архитектура: двунаправленный LSTM (2 слоя, hidden=128) + механизм Attention + два полносвязных слоя
- Данные: WLASL dataset (10 классов) + собственный датасет (10 классов, 60 примеров на слово)

---

## Структура проекта

```
HandTalker/
├── src/
│   ├── pytorch_transformer_train.py   # обучение модели
│   ├── realtime_tester.py             # тестирование: камера / видеофайл
│   ├── collect_dataset.py             # сбор собственного датасета
│   ├── wlasl_lstm_final.pt            # обученная модель (PyTorch)
│   ├── best_wlasl_lstm.pt             # лучший чекпоинт по val_acc
│   ├── label_classes.npy              # список распознаваемых классов
│   │
│   ├── services/                      # сервисный слой для FastAPI
│   │   ├── emotion_service.py         # анализ эмоций по видео
│   │   ├── sign_to_text_service.py    # жест -> текст
│   │   ├── tts_service.py             # текст -> WAV
│   │   └── audio_overlay_service.py   # наложение аудио на видео
│   │
│   ├── emotional_painting/
│   │   └── emotions_processor.py      # DeepFace-обёртка для анализа эмоций
│   │
│   ├── voice_generation/
│   │   └── voice_generator.py         # прямой запуск TTS (скрипт)
│   │
│   ├── my_dataset/                    # собственный датасет (10 слов x 60 MP4)
│   │   ├── accident/
│   │   ├── africa/
│   │   ├── apple/
│   │   └── ...
│   │
│   ├── wlasl_archive/                 # WLASL датасет
│   │   ├── WLASL_v0.3.json
│   │   └── videos/
│   │
│   ├── keypoints_cache/               # кэш кейпоинтов (NPY, генерируется автоматически)
│   └── mp_models/                     # модели MediaPipe (скачиваются автоматически)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd HandTalker
```

### 2. Создать виртуальное окружение

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

Основные зависимости:

| Пакет | Назначение |
|---|---|
| torch, torchvision | модель распознавания жестов |
| mediapipe | извлечение кейпоинтов (поза + руки) |
| opencv-python | работа с видео |
| deepface | анализ эмоций по лицу |
| TTS | синтез речи (Coqui TTS) |
| scikit-learn | LabelEncoder, train/test split |
| numpy, tqdm | вспомогательные утилиты |

### 4. Установить FFmpeg

Требуется для `AudioOverlayService`:

```bash
winget install ffmpeg
```

---

## Данные

### WLASL

Скачать датасет: [kaggle.com/datasets/risangbaskoro/wlasl-processed](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

Распаковать в:

```
src/wlasl_archive/
    WLASL_v0.3.json
    videos/
        00001.mp4
        00002.mp4
        ...
```

### Собственный датасет

Структура папки `my_dataset`:

```
src/my_dataset/
    <слово>/
        0.mp4
        1.mp4
        ...
        59.mp4
```

Для записи собственных жестов используйте `collect_dataset.py`.

---

## Обучение модели

```bash
cd src
python pytorch_transformer_train.py
```

Параметры в начале файла:

| Параметр | Значение | Описание |
|---|---|---|
| SUBSET | 10 | сколько слов взять из WLASL |
| SEQUENCE_LEN | 30 | кадров на жест |
| EPOCHS | 100 | максимум эпох |
| BATCH_SIZE | 32 | размер батча |
| LEARNING_RATE | 1e-3 | начальная скорость обучения |
| PATIENCE | 25 | early stopping |

Результат: `wlasl_lstm_final.pt` и `label_classes.npy` в папке `src/`.

Кейпоинты автоматически кэшируются в `keypoints_cache/`. Повторный запуск не перерабатывает уже обработанные видео.

---

## Тестирование в реальном времени

```bash
cd src

# Интерактивное меню
python realtime_tester.py

# Конкретный видеофайл
python realtime_tester.py --video path/to/video.mp4

# Без окна предпросмотра
python realtime_tester.py --video path/to/video.mp4 --no-window
```

Управление в режиме камеры:

| Клавиша | Действие |
|---|---|
| Q | выход, вывод итогового предложения |
| C | очистить буфер и предложение |
| S | сохранить предложение в sentence.txt |

---

## Сервисный слой

Все четыре сервиса находятся в `src/services/`. Каждый — отдельный класс, готовый к подключению в FastAPI-приложение.

### EmotionService

Принимает путь к видео, возвращает доминирующую эмоцию и усреднённые вероятности по всем эмоциям.

```python
from services.emotion_service import EmotionService

svc = EmotionService(sample_fps=1.0)
result = svc.analyze("video.mp4")
# result["dominant_emotion"]  ->  "happy"
# result["emotion_scores"]    ->  {"happy": 72.3, "sad": 5.1, ...}
```

### SignToTextService

Запускает LSTM-модель на видеофайле, возвращает строку-предложение без дубликатов подряд.

```python
from services.sign_to_text_service import SignToTextService

svc = SignToTextService()
sentence = svc.transcribe("video.mp4")
# sentence -> "apple bird basketball"
```

### TTSService

Принимает текст и эмоцию, генерирует WAV-файл. Темп речи автоматически подстраивается под эмоцию.

```python
from services.tts_service import TTSService

svc = TTSService(language="en")
wav_path = svc.synthesize(
    text="apple bird basketball",
    output_path="output/speech.wav",
    emotion="happy",
)
```

Сопоставление эмоции и темпа речи:

| Эмоция | Скорость |
|---|---|
| angry | 1.20 |
| happy | 1.15 |
| neutral | 1.00 |
| disgust | 0.90 |
| sad | 0.82 |

### AudioOverlayService

Накладывает TTS-аудио на видео через FFmpeg. Поддерживает смешивание с оригинальным звуком и полную замену дорожки.

```python
from services.audio_overlay_service import AudioOverlayService

svc = AudioOverlayService()

# Смешать TTS с оригинальным аудио
out = svc.overlay(
    video_path="input.mp4",
    audio_path="speech.wav",
    output_path="result.mp4",
    tts_volume=1.0,
    orig_volume=0.3,
)

# Полностью заменить аудио
out = svc.replace_audio("input.mp4", "speech.wav", "result.mp4")
```

## Результаты обучения

| Итерация | Top-1 | Top-5 | Изменения |
|---|---|---|---|
| 1 | 1.3% | 12.5% | базовая модель |
| 2 | 41.4% | 68.4% | убраны точки лица, мелкие правки |
| 3 | 81.6% | 91.1% | добавлен Attention + Gradient Clipping |
