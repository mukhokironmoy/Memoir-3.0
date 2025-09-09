import numpy as np
import torchaudio
from speechbrain.inference import SpeakerRecognition
from pydub import AudioSegment
import os
from V2T2 import v2t  # Import your voice-to-text function

# Lazy load SpeechBrain model
spk_verification = None
shashvat_embedding = np.load("shashvat_embeddings.npy").mean(axis=0)  # Load embedding once


def get_speaker_model():
    """Load SpeechBrain model only when needed."""
    global spk_verification
    if spk_verification is None:
        spk_verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model"
        )
    return spk_verification


def verify_speaker(segment_path):
    """Check if the speaker in the segment is Shashvat."""
    model = get_speaker_model()  # Load model only when needed
    signal, fs = torchaudio.load(segment_path)
    segment_embedding = model.encode_batch(signal).squeeze().detach().cpu().numpy()

    # Compute cosine similarity
    similarity = np.dot(shashvat_embedding, segment_embedding) / (
            np.linalg.norm(shashvat_embedding) * np.linalg.norm(segment_embedding)
    )

    return "Shashvat" if similarity > 0.75 else "Unknown"  # Threshold can be adjusted


def process_audio(file_path, output_folder):
    """Diarize, verify, and transcribe audio segments."""
    audio = AudioSegment.from_wav(file_path)
    duration = len(audio) / 1000  # Convert to seconds
    os.makedirs(output_folder, exist_ok=True)

    results = []

    # Split into fixed segments (e.g., 5 seconds each)
    for i, start in enumerate(range(0, int(duration), 5)):
        end = min(start + 5, duration)
        segment_audio = audio[start * 1000:end * 1000]
        segment_path = os.path.join(output_folder, f"segment_{i + 1}.wav")
        segment_audio.export(segment_path, format="wav")

        # Verify speaker identity
        speaker = verify_speaker(segment_path)

        # Transcribe speech
        language, text = v2t(segment_path)

         # Yield result immediately
        results.append(f"[{speaker}]: {text}")


    return results

