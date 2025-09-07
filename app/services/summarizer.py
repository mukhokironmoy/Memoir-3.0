from app.services.ai_client import load_gemini_client
from pathlib import Path
from google import genai
from google.genai import types
from app.logger import log

# setting up api client
client = load_gemini_client()

# Get file paths
DEFAULT_PROMPT = Path(r"app\static\data\default_prompt.txt")
DEFAULT_TRANSCRIPT = Path(r"app\static\data\test_transcript.txt")
DEFAULT_NOTES = Path(r"app\static\data\test_notes.txt")

def get_notes(prompt_path=DEFAULT_PROMPT, transcript_path=DEFAULT_TRANSCRIPT, notes_path=DEFAULT_NOTES):
    prompt_path = Path(prompt_path)
    transcript_path = Path(transcript_path)

    # prompt_first_line = f"Title = Test Notes \nVideo Title = {yt_data['Title']} \nUrl = {video_url}\n\n"

    sample_file = client.files.upload(file=transcript_path)

    with open(prompt_path, 'r', encoding="utf8") as f:
        prompt = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[sample_file, prompt],
        config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
    )

    with open(notes_path, 'w', encoding='utf8') as f:
        f.write(response.text)

    return notes_path

if __name__ == "__main__":
    log.info(f"Generating notes for DEFAULT_TRANSCRIPT : {DEFAULT_TRANSCRIPT}")
    get_notes()
    log.info(f"Notes Generated in DEFAULT_NOTES : {DEFAULT_NOTES}")
