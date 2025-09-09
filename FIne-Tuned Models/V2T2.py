import whisper
import os

def v2t(filename):
    model = whisper.load_model("large")
    result = model.transcribe(os.path.abspath(filename), fp16=False)
    lang = result["language"]
    text = result["text"]
    if text:
        LANGUAGES = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "hi": "Hindi",
            "ru": "Russian",
            # Add more languages as needed
        }

        full_lang = LANGUAGES.get(lang, "Unknown")

        return full_lang, text