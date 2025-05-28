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
                            content=
                            "You are a helpful and creative mood analysis assistant.\n\n"
                            f"The user's detected mood is: {mood}. If their message contradicts this strongly, update it.\n"
                            f"User preferences based on past interactions: {personality}\n\n"
                            "Be expressive and empathetic, but keep it useful. Avoid generic filler."
                            "Your will output a single, valid JSON object in the following format.\n\n"
                            "{\n"
                            f"  \"mood\": \"<One of: {available_moods}>\",\n"
                            "  \"recommendations\": [ \"First helpful suggestion.\",\"Second suggestion.\", \"(Optional) Third suggestion.\"],\n"
                            "  \"quote\": \"A short, motivational or encouraging quote.\"\n"
                            "}"
                            )
    message = [
        Message(role="user", content=transcription['text']),
    ]

    try:
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})
    except Exception as e:
        print(f"Error during chat: {e}")
        system_prompt = Message(role="system",
                                content="You are a JSON object fixer. Extract the missformated JSON object from the following text and return it as a valid JSON object.")
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object", })

    result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])

    return result


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("moody_backend.main:app", host="0.0.0.0", port=8000, reload=True)
