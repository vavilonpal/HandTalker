sign2text/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 data/
│   ├── raw/                # исходные видео с жестами
│   ├── processed/          # обработанные данные (CSV, keypoints)
│   └── labels.csv          # соответствие жест → текст
│
├── 📂 notebooks/
│   ├── data_preprocessing.ipynb   # подготовка данных
│   ├── model_training.ipynb       # обучение модели
│   └── evaluation.ipynb           # анализ результатов
│
├── 📂 src/
│   ├── __init__.py
│   │
│   ├── 📂 mediapipe_processor/
│   │   ├── __init__.py
│   │   ├── extract_keypoints.py   # извлечение 21 точки руки из видео
│   │   └── visualize_hands.py     # отрисовка точек на кадрах
│   │
│   ├── 📂 model/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py      # загрузка данных и формирование выборки
│   │   ├── train.py               # обучение нейросети
│   │   ├── evaluate.py            # тестирование модели
│   │   └── model.py               # архитектура сети (LSTM / Transformer)
│   │
│   ├── 📂 utils/
│   │   ├── __init__.py
│   │   ├── video_tools.py         # функции для чтения и резки видео
│   │   └── file_utils.py          # сохранение, логирование, пути и т.п.
│   │
│   ├── main.py                    # основной скрипт (распознавание жестов из видео/камеры)
│   └── predict.py                 # скрипт перевода жеста в текст
│
├── 📂 models/
│   ├── sign_model.pth             # сохранённая обученная модель (PyTorch)
│   └── label_encoder.pkl          # классы / слова
│
├── 📂 outputs/
│   ├── logs/                      # логи обучения
│   └── predictions/               # результаты тестов / видео с предсказаниями
│
└── 📂 app/
    ├── __init__.py
    ├── server.py                  # Flask/FastAPI сервер
    ├── routes.py                  # маршруты API
    └── templates/
        ├── index.html             # веб-интерфейс (если есть)
        └── static/
            ├── css/
            └── js/
