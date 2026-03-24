"""
EmotionService
==============
Wraps emotions_processor.py.
Accepts a video file path, returns the dominant emotion over the whole video
plus a per-second breakdown.

FastAPI usage example:
    from services.emotion_service import EmotionService

    svc = EmotionService()

    # In your endpoint:
    result = svc.analyze(video_path)
    # result = {
    #   "dominant_emotion": "happy",
    #   "emotion_scores":   {"happy": 72.3, "sad": 5.1, ...},
    #   "timeline": [{"timestamp": 0.0, "dominant_emotion": "happy", "emotions": {...}}, ...]
    # }
"""

import sys
import os
from collections import Counter

# Make sure emotions_processor is importable from src/emotional_painting/
_THIS_DIR   = os.path.dirname(__file__)
_SRC_DIR    = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_EMOTIONS_DIR = os.path.join(_SRC_DIR, "emotional_painting")
if _EMOTIONS_DIR not in sys.path:
    sys.path.insert(0, _EMOTIONS_DIR)

from emotions_processor import analyze_video  # type: ignore


class EmotionService:
    """
    Stateless service — safe to instantiate once and reuse across requests.
    """

    def __init__(self, sample_fps: float = 1.0):
        """
        Parameters
        ----------
        sample_fps : float
            How many frames per second to sample for emotion analysis.
            1.0 is a good default. Lower = faster; higher = more precise.
        """
        self.sample_fps = sample_fps

    # ------------------------------------------------------------------
    def analyze(self, video_path: str) -> dict:
        """
        Analyze emotion throughout the video.

        Parameters
        ----------
        video_path : str
            Absolute or relative path to the input video file.

        Returns
        -------
        dict with keys:
            dominant_emotion  – str  – emotion that appeared most frames
            emotion_scores    – dict – averaged probabilities per emotion (%)
            timeline          – list – raw per-frame records from emotions_processor
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        timeline: list[dict] = analyze_video(video_path, sample_fps=self.sample_fps)

        if not timeline:
            return {
                "dominant_emotion": "neutral",
                "emotion_scores":   {},
                "timeline":         [],
            }

        # ── Dominant emotion: majority vote across all sampled frames ──
        emotion_votes   = Counter(frame["dominant_emotion"] for frame in timeline)
        dominant        = emotion_votes.most_common(1)[0][0]

        # ── Average scores per emotion ──────────────────────────────────
        all_keys        = timeline[0]["emotions"].keys()
        emotion_scores  = {
            key: round(
                sum(frame["emotions"].get(key, 0.0) for frame in timeline) / len(timeline),
                2,
            )
            for key in all_keys
        }

        return {
            "dominant_emotion": dominant,
            "emotion_scores":   emotion_scores,
            "timeline":         timeline,
        }
