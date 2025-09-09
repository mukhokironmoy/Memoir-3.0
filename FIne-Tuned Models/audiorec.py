import pyaudio
import wave
import os
import threading
from datetime import datetime

import db

# Audio Configuration
mic = 1  # Adjust based on your input device
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
directory = "recordings"

os.makedirs(directory, exist_ok=True)

# Global variables
p = pyaudio.PyAudio()
stream = None
frames = []
filename = ""
recording = False  # Flag to control recording


def record():
    """Starts recording audio in a separate thread."""
    global stream, frames, filename, recording

    filename = os.path.join(directory, f"recording_{datetime.now().strftime('%d-%b%y_%H-%M-%S')}.wav")

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK, input_device_index=mic)

    frames = []
    recording = True
    print("Recording started...")

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)


def stop():
    """Stops recording and saves the audio file."""
    global stream, frames, filename, recording

    if stream:
        recording = False
        stream.stop_stream()
        stream.close()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recording stopped. Saved: {filename}")
    db.save_recording_info(filename)



def start_recording():
    """Starts recording in a new thread to prevent UI freezing."""
    global recording

    if not recording:  # Prevent multiple recordings at the same time
        threading.Thread(target=record, daemon=True).start()
