import json
from datetime import datetime
from typing import Optional

import dotenv

from moody_backend.GroqClient import GroqClient
from moody_backend.HfClient import HuggingfaceClient
from moody_backend.models import Message, AnalyzeResponse, AnalyzeResponseFastCheckin

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


available_moods = ["happy", "sad", "calm", "fearful", "angry", "disgust", "neutral", "surprised"]


@app.post("/emoji_checkin", response_model=AnalyzeResponseFastCheckin, operation_id="emoji_checkin",
          summary="Emoji mood check-in",
          description="Generate recommendations and a quote based on selected mood.")
async def emoji_checkin(mood: str = Form(...)):
    mood = mood.lower().strip()
    if mood not in available_moods:
        return {"error": f"Invalid mood: {mood}. Must be one of {available_moods}"}

    # Build system prompt
    system_prompt = Message(
        role="system",
        content=(
            "You are a helpful and creative mood analysis assistant.\n\n"
            f"- Allowed moods: {available_moods}\n"
            "- DO NOT invent or guess moods outside this list.\n"
            "- If unsure, keep the original mood.\n\n"
            "## Output format:\n"
            "Respond ONLY with a single valid JSON object in this format:\n"
            "{\n"
            "  \"mood\": \"<one of the allowed moods>\",\n"
            "  \"recommendations\": [\"First helpful suggestion.\", \"Second suggestion.\", \"(Optional) Third suggestion.\"],\n"
            "  \"quote\": \"A short, motivational or encouraging quote.\"\n"
            "}\n\n"
            "## Tone:\n"
            "- Be expressive and empathetic.\n"
            "- Keep it useful. Avoid generic filler like 'you got this!'."
            "- Use informal language (\"you\" instead of \"the user\"). Be personal, direct, warm and approachable.."
            "- Recommendations should be clear, actionable, and not phrased as a conversation. Avoid \"let's\" or asking questions."
        )
    )

    # LLM
    message = [Message(role="user", content=mood)]
    try:
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})
        result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error during chat: {e}")
        fallback_prompt = Message(
            role="system",
            content="You are a JSON object fixer. Extract the malformed JSON object from the following text and return it as a valid JSON object."
        )
        txt = txt2txtClient.chat(message, fallback_prompt, {"type": "json_object"})
        result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])

    # Check mood
    if result.get("mood") not in available_moods:
        print(f"Invalid mood '{result.get('mood')}' from LLM. Reverting to input mood: {mood}")
        result["mood"] = mood

    return result


@app.post("/analyze", response_model=AnalyzeResponse, operation_id="analyze_audio",
          summary="Analyze audio diary entry",
          description="Transcribe audio, detect mood, and update user persona based on the transcript.")
async def analyze_audio(audio: UploadFile, personality: Optional[str] = Form(None)):
    # 1. Transcribe and detect mood
    transcription = transcriptionClient.transcribe(audio.filename, audio.content_type, audio.file)
    transcript_text = transcription.to_dict()['text']
    emotions = moodClient.audio_classification(audio)
    detected_mood = emotions[0]['label']

    # 2. Update persona based on transcript and mood
    personality = update_persona(personality, transcript_text, detected_mood)

    # 3. Print debug info
    print(f"Transcription: {transcript_text}")
    print(f"Detected mood: {detected_mood}")
    print(f"Personality: {personality}")

    # 4. Build system prompt
    system_prompt = Message(
        role="system",
        content=(
            "You are a helpful and creative mood analysis assistant.\n\n"
            f"## Detected mood: '{detected_mood}'\n"
            "- Only update the mood if the user’s transcript strongly contradicts it.\n"
            f"- Allowed moods: {available_moods}\n"
            "- DO NOT invent or guess moods outside this list.\n"
            "- If unsure, keep the original mood.\n\n"
            "## User persona:\n"
            f"{json.dumps(personality, indent=2)}\n\n"
            "## Output format:\n"
            "Respond ONLY with a single valid JSON object in this format:\n"
            "{\n"
            "  \"mood\": \"<one of the allowed moods>\",\n"
            "  \"recommendations\": [\"First helpful suggestion.\", \"Second suggestion.\", \"(Optional) Third suggestion.\"],\n"
            "  \"quote\": \"A short, motivational or encouraging quote.\"\n"
            "}\n\n"
            "## Tone:\n"
            "- Be expressive and empathetic.\n"
            "- Keep it useful. Avoid generic filler like 'you got this!'."
            "- Use informal language (\"you\" instead of \"the user\"). Be personal, direct, warm and approachable.."
            "- Recommendations should be clear, actionable, and not phrased as a conversation. Avoid \"let's\" or asking questions."
        )
    )

    # 5. Get LLM response
    message = [Message(role="user", content=transcript_text)]
    try:
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})
        result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Error during chat: {e}")
        # Fallback fixer
        system_prompt = Message(
            role="system",
            content="You are a JSON object fixer. Extract the malformed JSON object from the following text and return it as a valid JSON object."
        )
        txt = txt2txtClient.chat(message, system_prompt, {"type": "json_object"})
        result = json.loads(txt.to_dict()["choices"][0]["message"]["content"])

    # 6. Validate and patch hallucinated mood if necessary
    if result.get("mood") not in available_moods:
        print(f"Invalid mood '{result.get('mood')}' from LLM. Reverting to original detected mood: {detected_mood}")
        result["mood"] = detected_mood

    # 7. Return final result
    result["personality"] = personality or {}
    result["transcription"] = transcript_text

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
