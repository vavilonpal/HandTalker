"""
TTSService
==========
Wraps Coqui TTS (tts_models/multilingual/multi-dataset/xtts_v2).
Accepts a text string + optional emotion hint, generates an audio WAV file.

FastAPI usage example:
    from services.tts_service import TTSService

    svc = TTSService()           # loads TTS model once (~2 GB, first call downloads)

    out_path = svc.synthesize(
        text="apple bird basketball",
        output_path="/tmp/speech.wav",
        emotion="happy",         # optional: adjusts speed
    )
    # out_path = "/tmp/speech.wav"
"""

import os
from TTS.api import TTS  # pip install TTS


# Speed modifiers per emotion (relative to 1.0 = neutral)
_EMOTION_SPEED: dict[str, float] = {
    "happy":    1.15,
    "sad":      0.82,
    "angry":    1.20,
    "fear":     1.10,
    "surprise": 1.10,
    "disgust":  0.90,
    "neutral":  1.00,
}

_DEFAULT_MODEL   = "tts_models/multilingual/multi-dataset/xtts_v2"
_DEFAULT_SPEAKER = "Ana Florence"
_DEFAULT_LANG    = "en"


class TTSService:
    """
    Lazily loads the TTS model on first call.

    Parameters
    ----------
    model_name : Coqui TTS model identifier
    speaker    : speaker voice name (xtts_v2 multi-speaker model)
    language   : BCP-47 language code, e.g. "en", "ru"
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        speaker:    str = _DEFAULT_SPEAKER,
        language:   str = _DEFAULT_LANG,
    ):
        self.model_name = model_name
        self.speaker    = speaker
        self.language   = language
        self._tts: TTS | None = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._tts is None:
            print(f"[TTSService] Loading model: {self.model_name} ...")
            self._tts = TTS(model_name=self.model_name)
            print("[TTSService] Model ready.")

    # ------------------------------------------------------------------
    def synthesize(
        self,
        text:        str,
        output_path: str,
        emotion:     str = "neutral",
    ) -> str:
        """
        Convert text to speech and save to output_path.

        Parameters
        ----------
        text        : Input sentence (e.g. "apple bird basketball")
        output_path : Where to write the WAV file (absolute or relative).
        emotion     : Hint from EmotionService to modulate speaking speed.
                      One of: happy / sad / angry / fear / surprise / disgust / neutral

        Returns
        -------
        str – absolute path to the generated WAV file.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        self._ensure_loaded()

        speed = _EMOTION_SPEED.get(emotion.lower(), 1.0)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        self._tts.tts_to_file(
            text=text,
            speaker=self.speaker,
            language=self.language,
            file_path=output_path,
            speed=speed,
        )

        return os.path.abspath(output_path)
