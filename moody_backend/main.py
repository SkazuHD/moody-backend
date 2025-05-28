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
txt2txtClient = GroqClient(dry_run=False)


@app.get("/")
async def root():
    return {"message": "Soullog API is Happily working! Are you?"}


@app.post("/analyze")
async def analyze(audio: UploadFile, personality: list = None):
    transcription = transcriptionClient.transcribe(audio.filename, audio.content_type, audio.file)

    # TODO Mood will be given by voice analysis
    # Or maybe combined with text analysis if voice does not match the text
    # mood = moodClient.analyze(audio.filename, audio.content_type, audio.file)

    # TODO use voice analysis to determine mood
    mood = "Happy"

    available_moods = 'Happy|Sad|Calm|Fear|Angry'

    system_prompt = Message(role="system",
                            content="BE CREATIVE! You are a mood analysis assistant. You will analyze the mood and just give recommendations based on the mood of the user. "
                                    f"Based on previous interactions we know that the user likes or dislikes the following ${personality}"
                                    f"Based on previous voice analysis the current mood of the user is ${mood} but you should consider if the text contradicts it heavily and adjust accordingly."
                                    f"Your Output should be in the form of a JSON object with the following keys: 'mood' : ${available_moods}, 'recommendations' : ['First', 'Second', 'Third if needed'].")

    message = [
        Message(role="user", content=transcription['text']),
    ]

    txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})

    result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])

    return result


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("moody_backend.main:app", host="0.0.0.0", port=8000, reload=True)
