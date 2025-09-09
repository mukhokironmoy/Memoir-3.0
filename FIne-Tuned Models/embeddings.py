from speechbrain.inference import EncoderClassifier
import torchaudio
import numpy as np
import torch
import os
from pydub import AudioSegment

# -----------------------------
# Load pretrained speaker embedding model from local dir
# -----------------------------
# First download manually with:
# huggingface-cli download speechbrain/spkrec-ecapa-voxceleb --local-dir ./models/spkrec-ecapa-voxceleb

embedding_model = EncoderClassifier.from_hparams(
    source="./models/spkrec-ecapa-voxceleb",   # local path instead of hub
    savedir="./models/spkrec-ecapa-voxceleb",  # make sure weights stay there
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)


def extract_embeddings(audio_path, chunk_size=5):
    """Extracts speaker embeddings from an audio file in fixed-length chunks."""

    audio = AudioSegment.from_wav(audio_path)
    duration = len(audio) / 1000  # Convert to seconds
    embeddings = []

    # Process in chunks if the file is longer than 30 seconds
    if duration > 30:
        for start in range(0, int(duration), chunk_size):
            end = min(start + chunk_size, duration)
            segment = audio[start * 1000:end * 1000]  # Convert to ms
            segment.export("temp_chunk.wav", format="wav")

            signal, fs = torchaudio.load("temp_chunk.wav")
            embedding = embedding_model.encode_batch(signal)
            embeddings.append(embedding.squeeze().detach().cpu().numpy())
    else:
        signal, fs = torchaudio.load(audio_path)
        embedding = embedding_model.encode_batch(signal)
        embeddings.append(embedding.squeeze().detach().cpu().numpy())

    return np.array(embeddings)


def process_speaker_embeddings(folder_path, output_file="shashvat_embeddings.npy"):
    """Extract embeddings from all WAV files in the speaker's folder."""
    all_embeddings = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            embeddings = extract_embeddings(file_path)
            all_embeddings.append(embeddings)

    np.save(output_file, np.vstack(all_embeddings))  # Save all embeddings
    print(f"Embeddings saved to {output_file}")


# Path to Shashvat’s recordings
shashvat_folder = "people/shashvat"
process_speaker_embeddings(shashvat_folder)
