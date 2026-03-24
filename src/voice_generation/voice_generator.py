from TTS.api import TTS

# Instantiate the TTS model (supports Russian and many other languages)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

text = "Мне очень грустно... Но потом я стал счастлив!"

tts.tts_to_file(
    text=text,
    speaker="Ana Florence",   # pick any available speaker voice
    language="ru",
    file_path="output.wav",
    speed=1  # медленнее = грустнее
)
