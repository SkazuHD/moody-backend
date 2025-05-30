import os

import torch
from fastapi import UploadFile
from transformers import pipeline
from pydub import AudioSegment
import io

SAFE_AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au"}

def convert_to_wav(file_bytes: bytes, format_hint: str) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=format_hint)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io.read()

class HuggingfaceClient:

    def __init__(self):
        #self.model = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
        self.model = "firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53"
        self.precision = torch.float32
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("audio-classification",
                             model=self.model,
                             torch_dtype=self.precision,
                             device=self.device)

    def audio_classification(self, audio: UploadFile):
        audio.file.seek(0)
        file_bytes = audio.file.read()
        ext = os.path.splitext(audio.filename)[1].lower()

        if ext not in SAFE_AUDIO_EXTS:
            format_hint = ext.lstrip('.')
            try:
                file_bytes = convert_to_wav(file_bytes, format_hint)
            except Exception as e:
                raise ValueError(f"Unsupported or unreadable audio format: {ext}") from e

        return self.pipe(file_bytes)