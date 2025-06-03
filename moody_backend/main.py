import json
from datetime import datetime
from typing import Optional

import dotenv

from moody_backend.GroqClient import GroqClient
from moody_backend.HfClient import HuggingfaceClient
from moody_backend.models import Message, AnalyzeResponse

dotenv.load_dotenv()

import uvicorn
from fastapi import FastAPI, UploadFile, Form

app = FastAPI()

transcriptionClient = GroqClient(dry_run=False)
moodClient = HuggingfaceClient()
txt2txtClient = GroqClient(dry_run=False)


@app.get("/", response_model=dict)
async def root():
    return {"message": "Soullog API is Happily working! Are you?",
            "date": datetime.isoformat(datetime.today())}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(audio: UploadFile, personality: Optional[str] = Form(None)):
    transcription = transcriptionClient.transcribe(audio.filename, audio.content_type, audio.file)
    emotions = moodClient.audio_classification(audio)
    mood = emotions[0]['label']

    personality = update_persona(personality, transcription.to_dict()['text'], mood)

    if True:
        print(f"Transcription: {transcription.to_dict()['text']}")
        print(f"Detected mood: {mood}")
        print(f"Personality: {personality}")

    available_moods = ["happy", "sad", "calm", "fearful", "angry", "disgust", "neutral", "suprised"]

    system_prompt = Message(
        role="system",
        content=(
            "You are a helpful and creative mood analysis assistant.\n\n"
            f"The detected mood for the current entry is: '{mood}'.\n"
            f"Based on the user's message and persona, update the mood ONLY if there is a strong contradiction.\n"
            f"Allowed moods are exactly: {available_moods}.\n"
            "DO NOT hallucinate or invent moods outside this list.\n"
            "If uncertain, keep the original mood.\n\n"
            f"User persona based on past interactions:\n{personality}\n\n"
            "Be expressive and empathetic, but keep it useful. Avoid generic filler.\n"
            "Your response MUST be a single valid JSON object exactly in this format:\n\n"
            "{\n"
            "  \"mood\": \"<one of the allowed moods>\",\n"
            "  \"recommendations\": [\"First helpful suggestion.\", \"Second suggestion.\", \"(Optional) Third suggestion.\"],\n"
            "  \"quote\": \"A short, motivational or encouraging quote.\"\n"
            "}"
        )
    )
    message = [
        Message(role="user", content=transcription.to_dict()['text']),
    ]
    print(system_prompt)
    try:
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})
    except Exception as e:
        print(f"Error during chat: {e}")
        system_prompt = Message(role="system",
                                content="You are a JSON object fixer. Extract the missformated JSON object from the following text and return it as a valid JSON object.")
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object", })

    result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])

    result["personality"] = personality if personality is not None else {}
    return result


def update_persona(personality, transcription: str, mood: str):
    system_prompt = """
    You are maintaining a user persona based on diary entries. You receive:
    1. The current persona (with metadata about recency of each item)
    2. A new diary transcript

    The persona has the following structure:
    {
      "long_term_traits": [string],
      "short_term_states": [{"state": string, "last_mentioned": int}],
      "contextual_insights": [{"insight": string, "last_mentioned": int}]
    }

    Your job is to update the persona based on the new transcript.

    Update logic:

    - For every short_term_state and contextual_insight:
      - Increment "last_mentioned" by 1
      - If it is mentioned again in the current transcript, reset "last_mentioned" to 0
      - If "last_mentioned" exceeds 3, remove the item from the persona

    - Add any new relevant short_term_states or contextual_insights from the transcript (with "last_mentioned": 0)

    - For long_term_traits:
      - Add traits if a consistent pattern or stable preference/value is described
      - Modify or remove only if the transcript shows a strong contradiction or change

    Avoid over-interpreting. Only make changes if something is clearly stated or strongly implied.

    Your response must return only the updated persona in valid JSON format with the exact same structure.
    """

    user_message = f"""
    Here is the current user persona and the latest voice diary transcript.

    Current persona:
    {personality}

    Transcript:
    \"\"\"{transcription}\"\"\"

    The detected mood for this entry is: {mood}

    Please update the persona based on the new transcript and mood, following the system instructions.
    Only return the updated persona JSON.
    """
    system = Message(role="system", content=system_prompt)
    message = [
        Message(role="user", content=user_message)
    ]

    response = txt2txtClient.chat(message, system, {"type": "json_object"})
    return json.loads(response.to_dict()["choices"][0]["message"]["content"])


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("moody_backend.main:app", host="0.0.0.0", port=8000, reload=True)
