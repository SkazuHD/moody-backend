import json

import dotenv

from moody_backend.GroqClient import GroqClient
from moody_backend.models import Message

dotenv.load_dotenv()

import uvicorn
from fastapi import FastAPI, UploadFile

app = FastAPI()

transcriptionClient = GroqClient(dry_run=True)
moodClient = "NOT IMPLEMENTED"
txt2txtClient = GroqClient(dry_run=True)


@app.get("/")
async def root():
    return {"message": "Soullog API is Happily working! Are you?"}


@app.post("/analyze")
async def analyze(audio: UploadFile):
    transcription = transcriptionClient.transcribe(audio.filename, audio.content_type, audio.file)

    # TODO Mood will be given by voice analysis
    # Or maybe combined with text analysis if voice does not match the text
    # mood = moodClient.analyze(audio.filename, audio.content_type, audio.file)

    # TODO change systempromt
    system_prompt = Message(role="system",
                            content="BE CREATIVE! You are a mood analysis assistant. You will analyze the mood and just give recommendations based on the mood of the user. Your Output should be in the form of a JSON object with the following keys: 'mood' : 'value', 'recommendations' : ['First', 'Second', 'Third if needed'].")

    message = [
        Message(role="user", content=transcription['text']),
    ]

    txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})

    result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])

    return result


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("moody_backend.main:app", host="0.0.0.0", port=8000, reload=True)
