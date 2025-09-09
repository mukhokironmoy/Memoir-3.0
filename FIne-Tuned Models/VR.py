from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import Segment
from pydub import AudioSegment
import os

# Load pretrained diarization model (replace with your model checkpoint if needed)
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization")


def diarize_audio(file_path, output_folder):
    """Diarize an audio file and store segments in a structured format."""

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Run diarization
    diarization = pipeline(file_path)

    # Load full audio
    audio = AudioSegment.from_wav(file_path)

    # Process each speaker segment
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_time = int(turn.start * 1000)  # Convert to milliseconds
        end_time = int(turn.end * 1000)
        segment_audio = audio[start_time:end_time]

        # Save segmented audio
        segment_filename = f"segment_{i + 1}.wav"
        segment_path = os.path.join(output_folder, segment_filename)
        segment_audio.export(segment_path, format="wav")
        print(f"Saved: {segment_path}")


# Main function to process all recordings
def process_all_recordings(recordings_folder, output_base_folder):
    """Diarize all WAV files in the recordings folder."""

    for file_name in os.listdir(recordings_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(recordings_folder, file_name)
            output_folder = os.path.join(output_base_folder, file_name.replace(".wav", ""))
            print(f"Processing: {file_name}")
            diarize_audio(file_path, output_folder)


# Paths (modify if needed)
recordings_folder = "recordings"  # Folder where original recordings are stored
output_base_folder = "processed_recordings"  # Where diarized segments will be stored

# Run the diarization process
process_all_recordings(recordings_folder, output_base_folder)
