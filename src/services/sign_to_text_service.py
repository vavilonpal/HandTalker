"""
SignToTextService
================
Wraps realtime_tester.analyze_video().
Accepts a video file path, runs the PyTorch LSTM sign-language model,
returns a sentence (deduplicated words, no consecutive duplicates).

FastAPI usage example:
    from services.sign_to_text_service import SignToTextService

    svc = SignToTextService()          # loads model once

    sentence = svc.transcribe(video_path)
    # sentence = "apple bird basketball"
"""

import os
import sys

# Make sure realtime_tester is importable from src/
_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR  = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from realtime_tester import analyze_video, load_model, SEQUENCE_LEN  # type: ignore


class SignToTextService:
    """
    Loads the PyTorch model once on first use (lazy) and reuses it for
    every subsequent call — no per-request overhead.

    Parameters
    ----------
    model_path   : path to wlasl_lstm_final.pt
    classes_path : path to label_classes.npy
    threshold    : minimum confidence to accept a prediction (0–1)
    show_window  : whether to show an OpenCV preview while processing
    """

    def __init__(
        self,
        model_path:   str   = None,
        classes_path: str   = None,
        threshold:    float = 0.45,
        show_window:  bool  = False,
    ):
        # Default paths relative to src/
        self.model_path   = model_path   or os.path.join(_SRC_DIR, "wlasl_lstm_final.pt")
        self.classes_path = classes_path or os.path.join(_SRC_DIR, "label_classes.npy")
        self.threshold    = threshold
        self.show_window  = show_window

        self._model   = None
        self._classes = None

    #
    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._classes = load_model(
                model_path=self.model_path,
                classes_path=self.classes_path,
            )

    #
    def transcribe(self, video_path: str) -> str:
        """
        Run sign-language recognition on the video and return a sentence.

        Parameters
        ----------
        video_path : str
            Path to the input video file.

        Returns
        -------
        str  –  space-separated words, e.g. "apple bird basketball"
                Empty string if nothing was recognised.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self._ensure_loaded()

        sentence = analyze_video(
            video_path=video_path,
            show_window=self.show_window,
        )
        return sentence
