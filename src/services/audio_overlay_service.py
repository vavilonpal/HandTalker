"""
AudioOverlayService
====================
Mixes a generated audio track onto the original video using FFmpeg.
The audio is stretched / trimmed to match the video duration exactly,
then mixed at a configurable volume over the original audio track.

FastAPI usage example:
    from services.audio_overlay_service import AudioOverlayService

    svc = AudioOverlayService()

    out_path = svc.overlay(
        video_path = "/tmp/input.mp4",
        audio_path = "/tmp/speech.wav",
        output_path= "/tmp/result.mp4",
        tts_volume = 1.0,       # 0.0 – 2.0
        orig_volume= 0.3,       # keep some original audio in the background
    )
    # out_path = "/tmp/result.mp4"

Requirements
------------
    FFmpeg must be installed and available on PATH:
        Windows: winget install ffmpeg   or   choco install ffmpeg
    Python:
        pip install ffmpeg-python
"""

import os
import subprocess
import shutil
import tempfile


def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "FFmpeg not found on PATH. "
            "Install it with: winget install ffmpeg  (or choco install ffmpeg)"
        )


class AudioOverlayService:
    """
    Overlays TTS-generated audio onto a video using FFmpeg subprocess calls.
    No Python-level audio processing — delegates everything to FFmpeg for
    maximum format compatibility and speed.
    """

    def __init__(self):
        _require_ffmpeg()

    # ------------------------------------------------------------------
    def overlay(
        self,
        video_path:  str,
        audio_path:  str,
        output_path: str,
        tts_volume:  float = 1.0,
        orig_volume: float = 0.0,
    ) -> str:
        """
        Mix TTS audio onto the original video.

        Parameters
        ----------
        video_path  : Path to the original input video.
        audio_path  : Path to the TTS WAV file (from TTSService).
        output_path : Where to write the final MP4.
        tts_volume  : Volume multiplier for the TTS track (default 1.0).
        orig_volume : Volume multiplier for the original video audio
                      (0.0 = mute original, 1.0 = keep at original level).

        Returns
        -------
        str – absolute path to output_path after successful encoding.

        Notes
        -----
        The TTS audio is looped if shorter than the video, and trimmed if
        longer, so the output always has the exact same duration as the input.
        """
        for p, label in [(video_path, "video"), (audio_path, "audio")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} file not found: {p}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # ── Get video duration ──────────────────────────────────────────
        duration = self._get_duration(video_path)

        # ── Build FFmpeg filter graph ───────────────────────────────────
        # We use amix to blend:
        #   [0:a] = original audio scaled by orig_volume
        #   [1:a] = TTS audio looped to fill video, scaled by tts_volume
        # The result is trimmed to video duration.
        filter_complex = (
            f"[0:a]volume={orig_volume}[orig];"
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=duration={duration},"
            f"volume={tts_volume}[tts];"
            f"[orig][tts]amix=inputs=2:duration=first:normalize=0[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,           # input 0: video (+ original audio)
            "-i", audio_path,           # input 1: TTS wav
            "-filter_complex", filter_complex,
            "-map", "0:v",              # video stream from input 0
            "-map", "[aout]",           # mixed audio
            "-c:v", "copy",             # no re-encoding of video
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    def replace_audio(
        self,
        video_path:  str,
        audio_path:  str,
        output_path: str,
    ) -> str:
        """
        Completely replace the video's audio track with the TTS audio.
        Simpler than overlay() — no mixing, just a stream copy of video
        plus the new audio track.

        Parameters
        ----------
        video_path  : Path to the original video.
        audio_path  : Path to the TTS WAV.
        output_path : Output MP4 path.

        Returns
        -------
        str – absolute path to output_path.
        """
        for p, label in [(video_path, "video"), (audio_path, "audio")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} file not found: {p}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        duration = self._get_duration(video_path)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            # Loop TTS if shorter than video; trim to video duration
            "-filter_complex",
            f"[1:a]aloop=loop=-1:size=2e+09,atrim=duration={duration}[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
            )

        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_duration(video_path: str) -> float:
        """Return video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 60.0  # fallback
