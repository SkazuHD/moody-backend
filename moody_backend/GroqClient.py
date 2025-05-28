import os

from groq import Groq


class MockedChatResponse:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class GroqClient:
    mocked_transcription = {
        "text": "My dog died, i cant do this anymore",
        "x_groq": {
            "id": "req_mocked1234567890"
        }
    }

    mocked_chat_response = MockedChatResponse({
        "id": "chatcmpl-8efb3e29-17a1-4eff-a8c3-39cdba4c0b88",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": (
                        '{\n'
                        '  "mood": "overwhelmed",\n'
                        '  "recommendations": [\n'
                        '    "Take a deep breath and allow yourself to grieve in your own time."',
                        '    "Consider reaching out to a trusted friend or family member for support."',
                        '    "You might also find it helpful to create a memory book or cherished keepsakes related to your dog."'
                        '  ]\n'
                        '}'
                    ), "role": "assistant",
                }
            }
        ],
        "created": 1748427286,
        "model": "llama-3.1-8b-instant",
        "object": "chat.completion",
        "system_fingerprint": "fp_a4265e44d5",
        "usage": {
            "completion_tokens": 8,
            "prompt_tokens": 42,
            "total_tokens": 50,
            "completion_time": 0.010666667,
            "prompt_time": 0.002446889,
            "queue_time": 0.092985285,
            "total_time": 0.013113556
        },
        "x_groq": {
            "id": "req_01jwb66a8fexms0nmx4wsk91zc"
        }
    })

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY", "your-groq-api-key"))

    def transcribe(self, filename, content_type, file):
        if self.dry_run:
            return self.mocked_transcription

        transcription = self.client.audio.transcriptions.create(
            file=(filename, file, content_type),
            model="whisper-large-v3-turbo",
        )

        return transcription

    def chat(self, messages: list, system_prompt=None, response_format=None):
        if self.dry_run:
            return self.mocked_chat_response

        if system_prompt:
            messages.insert(0, system_prompt)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_completion_tokens=300,
            top_p=1,
            stream=False,
            response_format=response_format,
            stop=None,
        )

        return response
