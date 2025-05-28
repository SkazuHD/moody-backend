import torch
from fastapi import UploadFile
from transformers import pipeline


class HugginfaceClient:

    def __init__(self):
        self.precision = torch.float32
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("audio-classification",
                             model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
                             torch_dtype=self.precision, device=self.device)

    def audio_classification(self, audio: UploadFile):
        audio.file.seek(0)
        audio_bytes = audio.file.read()

        emotions = self.pipe(audio_bytes)
        return emotions
