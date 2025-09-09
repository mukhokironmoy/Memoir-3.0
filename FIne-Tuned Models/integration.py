import os
import numpy as np
import json
import pickle
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pydub import AudioSegment
import torchaudio
from speechbrain.inference import SpeakerRecognition, EncoderClassifier
from V2T2 import v2t


class ConversationProcessor:
    def __init__(self, embeddings_db_path="speaker_embeddings.pkl", threshold=0.75):
        """Initialize the conversation processor."""
        # Load diarization pipeline
        self.diarization_pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization")

        # Lazy load models
        self.spk_verification = None
        self.embedding_model = None

        # Speaker embeddings database
        self.embeddings_db_path = embeddings_db_path
        self.speaker_embeddings = self.load_embeddings_db()
        self.threshold = threshold

    def get_speaker_model(self):
        """Load SpeechBrain speaker verification model only when needed."""
        if self.spk_verification is None:
            self.spk_verification = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model"
            )
        return self.spk_verification

    def get_embedding_model(self):
        """Load SpeechBrain embedding model only when needed."""
        if self.embedding_model is None:
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="tmp_model",
                run_opts={"fetch_strategy": "copy"}
            )
        return self.embedding_model

    def load_embeddings_db(self):
        """Load the speaker embeddings database."""
        if os.path.exists(self.embeddings_db_path):
            with open(self.embeddings_db_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_embeddings_db(self):
        """Save the speaker embeddings database."""
        with open(self.embeddings_db_path, 'wb') as f:
            pickle.dump(self.speaker_embeddings, f)

    def extract_embeddings_from_audio(self, audio_path, chunk_size=5):
        """Extract speaker embeddings from an audio file in fixed-length chunks."""
        model = self.get_embedding_model()

        audio = AudioSegment.from_wav(audio_path)
        duration = len(audio) / 1000  # Convert to seconds
        embeddings = []

        # Process in chunks if the file is longer than 30 seconds
        if duration > 30:
            temp_chunk_path = "temp_chunk_extract.wav"
            try:
                for start in range(0, int(duration), chunk_size):
                    end = min(start + chunk_size, duration)
                    segment = audio[start * 1000:end * 1000]  # Convert to ms
                    segment.export(temp_chunk_path, format="wav")

                    signal, fs = torchaudio.load(temp_chunk_path)
                    embedding = model.encode_batch(signal)
                    embeddings.append(embedding.squeeze().detach().cpu().numpy())
            finally:
                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)
        else:
            signal, fs = torchaudio.load(audio_path)
            embedding = model.encode_batch(signal)
            embeddings.append(embedding.squeeze().detach().cpu().numpy())

        return np.array(embeddings)

    def add_speaker(self, speaker_name, audio_files_or_folder):
        """Add a new speaker to the database by processing their audio files."""
        print(f"Processing speaker: {speaker_name}")

        all_embeddings = []
        audio_files = []

        # Handle both single files and folders
        if isinstance(audio_files_or_folder, str):
            if os.path.isfile(audio_files_or_folder):
                # Single file
                audio_files = [audio_files_or_folder]
            elif os.path.isdir(audio_files_or_folder):
                # Folder - get all WAV files
                for file_name in os.listdir(audio_files_or_folder):
                    if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                        audio_files.append(os.path.join(audio_files_or_folder, file_name))
        elif isinstance(audio_files_or_folder, list):
            # List of files
            audio_files = audio_files_or_folder

        if not audio_files:
            print(f"No audio files found for {speaker_name}")
            return False

        # Extract embeddings from all files
        for audio_file in audio_files:
            if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                continue

            print(f"  Processing: {os.path.basename(audio_file)}")
            try:
                # Convert to WAV if needed
                temp_wav = None
                if not audio_file.lower().endswith('.wav'):
                    temp_wav = "temp_convert.wav"
                    audio = AudioSegment.from_file(audio_file)
                    audio.export(temp_wav, format="wav")
                    process_file = temp_wav
                else:
                    process_file = audio_file

                embeddings = self.extract_embeddings_from_audio(process_file)
                all_embeddings.extend(embeddings)

                # Clean up temp file
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)

            except Exception as e:
                print(f"    Error processing {audio_file}: {e}")
                continue

        if all_embeddings:
            # Calculate mean embedding for the speaker
            speaker_embedding = np.array(all_embeddings).mean(axis=0)
            self.speaker_embeddings[speaker_name] = speaker_embedding
            self.save_embeddings_db()
            print(f"  Added {speaker_name} with {len(all_embeddings)} embeddings")
            return True
        else:
            print(f"  No valid embeddings extracted for {speaker_name}")
            return False

    def identify_speaker(self, segment_path):
        """Identify the speaker in an audio segment."""
        if not self.speaker_embeddings:
            return "Unknown"

        try:
            model = self.get_speaker_model()
            signal, fs = torchaudio.load(segment_path)
            segment_embedding = model.encode_batch(signal).squeeze().detach().cpu().numpy()

            best_match = "Unknown"
            best_similarity = 0

            # Compare with all known speakers
            for speaker_name, speaker_embedding in self.speaker_embeddings.items():
                # Compute cosine similarity
                similarity = np.dot(speaker_embedding, segment_embedding) / (
                        np.linalg.norm(speaker_embedding) * np.linalg.norm(segment_embedding)
                )

                if similarity > best_similarity and similarity > self.threshold:
                    best_similarity = similarity
                    best_match = speaker_name

            return best_match

        except Exception as e:
            print(f"Error identifying speaker: {e}")
            return "Unknown"

    def list_known_speakers(self):
        """List all known speakers in the database."""
        if not self.speaker_embeddings:
            print("No speakers in the database.")
            return []

        print("Known speakers:")
        for speaker in self.speaker_embeddings.keys():
            print(f"  - {speaker}")
        return list(self.speaker_embeddings.keys())

    def remove_speaker(self, speaker_name):
        """Remove a speaker from the database."""
        if speaker_name in self.speaker_embeddings:
            del self.speaker_embeddings[speaker_name]
            self.save_embeddings_db()
            print(f"Removed speaker: {speaker_name}")
            return True
        else:
            print(f"Speaker {speaker_name} not found in database.")
            return False

    def diarize_and_process(self, file_path, temp_folder="temp_segments"):
        """Diarize audio and return segments with speaker labels and timestamps."""
        # Ensure temp directory exists
        os.makedirs(temp_folder, exist_ok=True)

        # Run diarization
        print("Running diarization...")
        diarization = self.diarization_pipeline(file_path)

        # Load full audio
        audio = AudioSegment.from_wav(file_path)

        segments = []

        # Process each speaker segment
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            start_time = turn.start
            end_time = turn.end

            # Extract segment
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment_audio = audio[start_ms:end_ms]

            # Save temporary segment file
            segment_filename = f"temp_segment_{i}.wav"
            segment_path = os.path.join(temp_folder, segment_filename)
            segment_audio.export(segment_path, format="wav")

            # Identify speaker
            identified_speaker = self.identify_speaker(segment_path)

            # Transcribe the segment
            print(f"Transcribing segment {i + 1}...")
            try:
                language, text = v2t(segment_path)
                if text:
                    text = text.strip()
                else:
                    text = "[No speech detected]"
            except Exception as e:
                print(f"Error transcribing segment {i + 1}: {e}")
                text = "[Transcription failed]"
                language = "Unknown"

            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'speaker': identified_speaker,
                'original_speaker': speaker,
                'text': text,
                'language': language
            })

            # Clean up temp file
            os.remove(segment_path)

        # Clean up temp directory
        if os.path.exists(temp_folder) and not os.listdir(temp_folder):
            os.rmdir(temp_folder)

        return segments

    def format_conversation(self, segments):
        """Format the conversation segments into readable text."""
        conversation_text = []
        conversation_text.append("=== CONVERSATION TRANSCRIPT ===\n")

        # Add speaker summary
        speakers = set(seg['speaker'] for seg in segments if seg['speaker'] != 'Unknown')
        if speakers:
            conversation_text.append(f"Known Speakers: {', '.join(speakers)}\n")
        conversation_text.append("\n")

        current_speaker = None
        current_text_block = []

        for segment in segments:
            speaker = segment['speaker']
            text = segment['text']
            start_time = segment['start_time']
            end_time = segment['end_time']

            # Format timestamp
            def format_time(seconds):
                minutes = int(seconds // 60)
                seconds = int(seconds % 60)
                return f"{minutes:02d}:{seconds:02d}"

            timestamp = f"[{format_time(start_time)} - {format_time(end_time)}]"

            # If speaker changed or this is the first segment
            if speaker != current_speaker:
                # Finish previous speaker's block if exists
                if current_speaker is not None and current_text_block:
                    conversation_text.append(" ".join(current_text_block) + "\n\n")

                # Start new speaker block
                current_speaker = speaker
                current_text_block = []
                conversation_text.append(f"{speaker} {timestamp}: ")

            # Add text to current block (skip empty or failed transcriptions)
            if text and text not in ["[No speech detected]", "[Transcription failed]"]:
                current_text_block.append(text)

        # Don't forget the last speaker's block
        if current_text_block:
            conversation_text.append(" ".join(current_text_block) + "\n\n")

        return "".join(conversation_text)

    def process_conversation(self, wav_file_path, output_txt_path=None):
        """Main function to process a WAV file and create a conversation transcript."""
        if not os.path.exists(wav_file_path):
            print(f"Error: File {wav_file_path} not found.")
            return None

        print(f"Processing conversation: {wav_file_path}")

        # Diarize and process segments
        segments = self.diarize_and_process(wav_file_path)

        if not segments:
            print("No segments found in the audio file.")
            return None

        # Format conversation
        conversation_text = self.format_conversation(segments)

        # Generate output filename if not provided
        if output_txt_path is None:
            base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
            output_txt_path = f"{base_name}_transcript.txt"

        # Save to file
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(conversation_text)

        print(f"\nTranscript saved to: {output_txt_path}")
        print("\n" + "=" * 50)
        print("CONVERSATION PREVIEW:")
        print("=" * 50)
        print(conversation_text[:1000] + "..." if len(conversation_text) > 1000 else conversation_text)

        return output_txt_path


def main():
    """Main function with command-line interface."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Process conversation: python main_processor.py process <wav_file> [output_txt]")
        print("  Add speaker:         python main_processor.py add_speaker <name> <audio_folder_or_file>")
        print("  List speakers:       python main_processor.py list_speakers")
        print("  Remove speaker:      python main_processor.py remove_speaker <name>")
        print("\nExamples:")
        print("  python main_processor.py add_speaker John ./john_recordings/")
        print("  python main_processor.py add_speaker Mary mary_sample.wav")
        print("  python main_processor.py process conversation.wav")
        return

    command = sys.argv[1].lower()
    processor = ConversationProcessor()

    if command == "process":
        if len(sys.argv) < 3:
            print("Error: WAV file path required for processing.")
            return
        wav_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        processor.process_conversation(wav_file, output_file)

    elif command == "add_speaker":
        if len(sys.argv) < 4:
            print("Error: Speaker name and audio path required.")
            return
        speaker_name = sys.argv[2]
        audio_path = sys.argv[3]
        success = processor.add_speaker(speaker_name, audio_path)
        if success:
            print(f"Successfully added speaker: {speaker_name}")
        else:
            print(f"Failed to add speaker: {speaker_name}")

    elif command == "list_speakers":
        processor.list_known_speakers()

    elif command == "remove_speaker":
        if len(sys.argv) < 3:
            print("Error: Speaker name required.")
            return
        speaker_name = sys.argv[2]
        processor.remove_speaker(speaker_name)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()