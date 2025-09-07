from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from pathlib import Path

def load_gemini_client():
    # API Setup
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    return client

