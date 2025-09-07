# main.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import sounddevice as sd
import soundfile as sf

# Gemini SDK
from google import genai
from google.genai import types

# ---------------- Fixed locations ----------------
AUDIO_OUT_PATH = Path("recordings/input.wav").resolve()
TRANSCRIPT_OUT_PATH = Path("outputs/transcript.txt").resolve()
NOTES_OUT_PATH = Path("outputs/notes.txt").resolve()
PROMPT_PATH = Path("default_prompt.txt").resolve()

AUDIO_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash"


# ---------------- Helpers ----------------
def _load_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    return genai.Client(api_key=api_key)


# ---------------- Recording backend ----------------
class InteractiveRecorder:
    def __init__(self, samplerate: int = 16_000, channels: int = 1, dtype: str = "int16"):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._is_open = False
        self._is_recording = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[REC][warn] {status}", file=sys.stderr)
        if self._is_recording:
            self._frames.append(indata.copy())

    def open(self):
        if self._is_open:
            return
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()
        self._is_open = True
        print("[REC] Stream opened.")

    def start(self):
        if not self._is_open:
            self.open()
        self._is_recording = True
        print("[REC] Recording started.")

    def pause(self):
        if self._is_open and self._is_recording:
            self._is_recording = False
            print("[REC] Paused.")

    def resume(self):
        if self._is_open and not self._is_recording:
            self._is_recording = True
            print("[REC] Resumed.")

    def stop_and_save(self, out_path: Path) -> Path:
        if self._is_open:
            self._is_recording = False
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
                self._is_open = False
            print("[REC] Stream closed.")

        if len(self._frames) == 0:
            print("[REC] No audio captured. Creating empty file.")
            audio = np.zeros((0, self.channels), dtype=np.int16)
        else:
            audio = np.concatenate(self._frames, axis=0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, self.samplerate, subtype="PCM_16")
        print(f"[REC] Saved: {out_path}  ({audio.shape[0]/self.samplerate:.2f}s)")
        self._frames.clear()
        return out_path


def record_audio_interactive(out_path: Path) -> Path:
    rec = InteractiveRecorder()
    print("\n=== Microphone Recorder ===")
    print("Controls:  S=Start, P=Pause, R=Resume, T=Stop&Save, Q=Quit\n")

    saved_path: Path | None = None
    while True:
        try:
            cmd = input("[S/P/R/T/Q] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[REC] Exiting.")
            break

        if cmd == "s":
            rec.start()
        elif cmd == "p":
            rec.pause()
        elif cmd == "r":
            rec.resume()
        elif cmd == "t":
            saved_path = rec.stop_and_save(out_path)
            break
        elif cmd == "q":
            print("[REC] Quit requested.")
            sys.exit(0)
        else:
            print("Unknown command.")
    return saved_path or out_path


# ---------------- Transcription ----------------
def transcribe_with_gemini_files_api(
    audio_path: Path,
    transcript_out_path: Path,
    speaker1: str,
    speaker2: str,
    model_name: str = MODEL_NAME,
    prompt: str = "Transcribe this audio verbatim with punctuation.",
) -> Path:
    client = _load_gemini_client()

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"Audio file missing or empty: {audio_path}")

    print(f"[GEMINI] Uploading: {audio_path}")
    uploaded = client.files.upload(file=str(audio_path))

    full_prompt = (
        f"{prompt}\n\n"
        f"Use '{speaker1}' and '{speaker2}' as speaker labels if diarization is possible."
    )
    print(f"[GEMINI] Generating transcript with model: {model_name}")
    response = client.models.generate_content(
        model=model_name,
        contents=[full_prompt, uploaded],
        # Keep config minimal for broad model compatibility
        config=types.GenerateContentConfig(),
    )

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Empty response from Gemini.")

    transcript_out_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_out_path.write_text(text, encoding="utf-8")
    print(f"[GEMINI] Transcript written to: {transcript_out_path}")
    return transcript_out_path


# ---------------- Notes generation ----------------
def generate_notes_from_transcript(
    transcript_path: Path,
    prompt_path: Path,
    notes_out_path: Path,
    model_name: str = MODEL_NAME,
) -> Path:
    client = _load_gemini_client()

    if not transcript_path.exists() or transcript_path.stat().st_size == 0:
        raise FileNotFoundError(f"Transcript missing or empty: {transcript_path}")

    if not prompt_path.exists() or prompt_path.stat().st_size == 0:
        raise FileNotFoundError(f"Prompt file missing or empty: {prompt_path}")

    print(f"[GEMINI] Uploading transcript: {transcript_path}")
    transcript_file = client.files.upload(file=str(transcript_path))

    prompt_text = prompt_path.read_text(encoding="utf-8")

    print(f"[GEMINI] Generating notes with model: {model_name}")
    response = client.models.generate_content(
        model=model_name,
        contents=[transcript_file, prompt_text],
        config=types.GenerateContentConfig(),
    )

    notes_text = (getattr(response, "text", "") or "").strip()
    if not notes_text:
        raise RuntimeError("Empty notes response from Gemini.")

    notes_out_path.parent.mkdir(parents=True, exist_ok=True)
    notes_out_path.write_text(notes_text, encoding="utf-8")
    print(f"[GEMINI] Notes written to: {notes_out_path}")
    return notes_out_path


# ---------------- Script-style main ----------------
def main():
    print(">>> Enter speaker names first.")
    speaker1 = input("Name of Speaker 1: ").strip() or "Speaker 1"
    speaker2 = input("Name of Speaker 2: ").strip() or "Speaker 2"

    # --- Step 1: Record (optional) ---
    print("\n>>> Step 1: Record audio")
    use_recorder = input("Record new audio now? [y/N]: ").strip().lower() == "y"
    if use_recorder:
        audio_path = record_audio_interactive(AUDIO_OUT_PATH)
    else:
        audio_path = AUDIO_OUT_PATH
        print(f"[INFO] Using existing audio at: {audio_path}")

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        print("[REC] No audio to transcribe. Exiting.")
        print("\nAll done. Bye!")
        return

    # --- Step 2: Transcribe ---
    ans = input("\n>>> Step 2: Transcribe with Gemini? [y/N]: ").strip().lower()
    if ans == "y":
        try:
            transcribe_with_gemini_files_api(audio_path, TRANSCRIPT_OUT_PATH, speaker1, speaker2)
        except Exception as e:
            print(f"[ERR] Transcription failed: {e}")
            print("\nAll done. Bye!")
            return
    else:
        print("[INFO] Skipped transcription (will look for existing transcript).")

    # Verify transcript exists
    if not TRANSCRIPT_OUT_PATH.exists() or TRANSCRIPT_OUT_PATH.stat().st_size == 0:
        print(f"[WARN] Transcript not found at {TRANSCRIPT_OUT_PATH}. Cannot generate notes.")
        print("\nAll done. Bye!")
        return

    # --- Step 3: Notes generation prompt (your new requirement) ---
    print("\n>>> Step 3: Generate notes from the transcript")
    make_notes = input("Generate notes now? [y/N]: ").strip().lower() == "y"
    if make_notes:
        # Let user override prompt and output paths if they want
        prompt_in = input(f"Path to prompt file [{PROMPT_PATH}]: ").strip()
        notes_out_in = input(f"Path to save notes [{NOTES_OUT_PATH}]: ").strip()

        prompt_path = Path(prompt_in) if prompt_in else PROMPT_PATH
        notes_out_path = Path(notes_out_in) if notes_out_in else NOTES_OUT_PATH

        try:
            outp = generate_notes_from_transcript(
                transcript_path=TRANSCRIPT_OUT_PATH,
                prompt_path=prompt_path,
                notes_out_path=notes_out_path,
            )
            print(f"\n[OK] Notes saved to: {outp}")
        except Exception as e:
            print(f"[ERR] Notes generation failed: {e}")
    else:
        print("Skipped notes generation.")

    print("\nAll done. Bye!")


if __name__ == "__main__":
    main()
