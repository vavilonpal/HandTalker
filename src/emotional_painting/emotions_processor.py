import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from deepface import DeepFace
from pathlib import Path


# Конфигурация

VIDEO_PATH  = "happy-video.mp4"
SAMPLE_FPS  = 1.0                 # кадров в секунду для анализа (0.5 / 1 / 2)
OUTPUT_DIR  = "output"            # папка для результатов

EMOTION_COLORS = {
    "happy":    "#EF9F27",
    "sad":      "#3B8BD4",
    "angry":    "#E24B4A",
    "fear":     "#D85A30",
    "surprise": "#1D9E75",
    "neutral":  "#888780",
    "disgust":  "#7F77DD",
}


# JSON-энкодер для numpy-типов

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Анализ видео

def analyze_video(video_path: str, sample_fps: float = 1.0) -> list[dict]:
    """
    Извлекает кадры из видео с заданной частотой и анализирует эмоции.

    Параметры:
        video_path  — путь к видеофайлу
        sample_fps  — сколько кадров в секунду анализировать

    Возвращает список словарей:
        {timestamp, dominant_emotion, emotions: {happy, sad, ...}}
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    video_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration      = total_frames / video_fps
    frame_interval = max(1, int(video_fps / sample_fps))

    print(f"  Длительность : {duration:.1f} сек")
    print(f"  FPS видео    : {video_fps:.1f}")
    print(f"  Анализируем  : каждый {frame_interval}-й кадр ({sample_fps} fps)\n")

    results    = []
    frame_idx  = 0
    analyzed   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = round(frame_idx / video_fps, 2)

            try:
                analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                # DeepFace может вернуть список (несколько лиц)
                face = analysis[0] if isinstance(analysis, list) else analysis

                # Конвертируем все значения в чистый Python float
                emotions  = {k: round(float(v), 2) for k, v in face["emotion"].items()}
                dominant  = str(face["dominant_emotion"])

            except Exception as e:
                print(f"  [!] {timestamp}s — лицо не найдено ({e})")
                emotions = {e: 0.0 for e in EMOTION_COLORS}
                emotions["neutral"] = 100.0
                dominant = "neutral"

            results.append({
                "timestamp":        timestamp,
                "dominant_emotion": dominant,
                "emotions":         emotions,
            })

            analyzed += 1
            bar = "*" * int(20 * frame_idx / total_frames) + "*" * (20 - int(20 * frame_idx / total_frames))
            print(f"  [{bar}] {timestamp:.1f}s → {dominant}", end="\r")

        frame_idx += 1

    cap.release()
    print(f"\n\n  Готово: проанализировано {analyzed} кадров.")
    return results


# Сохранение результатов

def save_results(results: list[dict], output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """Сохраняет results в JSON и CSV, возвращает DataFrame."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # JSON — с кастомным энкодером, безопасно для numpy
    json_path = f"{output_dir}/emotions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"  JSON сохранён : {json_path}")

    # CSV — плоская таблица timestamp + все эмоции
    rows = []
    for r in results:
        row = {"timestamp": r["timestamp"], "dominant": r["dominant_emotion"]}
        row.update(r["emotions"])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/emotions.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV сохранён  : {csv_path}")

    return df


# Построение графика

def plot_timeline(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """
    Строит двухуровневый график:
      - верхняя полоска — доминирующая эмоция (цветная)
      - нижний график  — стекированные вероятности всех эмоций
    """
    emotions_in_df = [e for e in EMOTION_COLORS if e in df.columns]
    timestamps     = df["timestamp"].tolist()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 7),
        gridspec_kw={"height_ratios": [1, 4]},
    )
    fig.patch.set_facecolor("#0f0f0f")

    # Верхняя полоска: доминирующая эмоция
    ax1 = axes[0]
    ax1.set_facecolor("#0f0f0f")

    for i in range(len(timestamps) - 1):
        emotion = df["dominant"].iloc[i]
        color   = EMOTION_COLORS.get(emotion, "#888780")
        width   = timestamps[i + 1] - timestamps[i]
        ax1.barh(0, width, left=timestamps[i], color=color, height=1, linewidth=0)

    # Последний сегмент
    last_emotion = df["dominant"].iloc[-1]
    ax1.barh(0, 0.5, left=timestamps[-1],
             color=EMOTION_COLORS.get(last_emotion, "#888780"), height=1, linewidth=0)

    ax1.set_xlim(timestamps[0], timestamps[-1] + 0.5)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_title("Emotion Timeline", color="#e0e0e0", fontsize=13,
                  fontweight="bold", pad=10, loc="left")

    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Легенда
    patches = [
        mpatches.Patch(color=c, label=e)
        for e, c in EMOTION_COLORS.items()
    ]
    ax1.legend(
        handles=patches,
        loc="upper right",
        facecolor="#1a1a1a",
        labelcolor="#cccccc",
        edgecolor="#333",
        fontsize=9,
        ncol=7,
    )

    # Нижний график: стекированные вероятности
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f0f")

    ax2.stackplot(
        df["timestamp"],
        [df[e] for e in emotions_in_df],
        labels=emotions_in_df,
        colors=[EMOTION_COLORS[e] for e in emotions_in_df],
        alpha=0.85,
    )

    ax2.set_xlim(timestamps[0], timestamps[-1] + 0.5)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Время (сек)", color="#999", fontsize=11)
    ax2.set_ylabel("Вероятность (%)", color="#999", fontsize=11)
    ax2.tick_params(colors="#888")
    ax2.grid(axis="y", color="#222", linewidth=0.8)

    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")

    ax2.legend(
        loc="upper right",
        facecolor="#1a1a1a",
        labelcolor="#cccccc",
        edgecolor="#333",
        fontsize=9,
        ncol=2,
    )

    plt.tight_layout(h_pad=0.3)

    png_path = f"{output_dir}/emotion_timeline.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  График сохранён: {png_path}")


# Краткая сводка

def print_summary(df: pd.DataFrame):
    """Выводит краткую статистику по видео."""
    total = len(df)
    print("\n  ── Сводка ──────────────────────────────────")
    counts = df["dominant"].value_counts()
    for emotion, count in counts.items():
        pct   = count / total * 100
        bar   = "*" * int(pct / 5)
        color_name = emotion.ljust(10)
        print(f"  {color_name} {bar:<20} {pct:5.1f}%")
    print(f"\n  Длительность  : {df['timestamp'].iloc[-1]:.1f} сек")
    print(f"  Точек данных  : {total}")
    print("  ────────────────────────────────────────────")


# Точка входа

if __name__ == "__main__":
    print(f"\n{'─'*50}")
    print(f"  Анализируем: {VIDEO_PATH}")
    print(f"{'─'*50}\n")

    results = analyze_video(VIDEO_PATH, sample_fps=SAMPLE_FPS)

    print("\nСохраняем результаты...")
    df = save_results(results)

    print("\nСтроим график...")
    plot_timeline(df)

    print_summary(df)

    print(f"\nВсё готово. Результаты в папке: {OUTPUT_DIR}/\n")