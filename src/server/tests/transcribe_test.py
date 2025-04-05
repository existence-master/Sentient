import pyaudio
import numpy as np
import time
from faster_whisper import WhisperModel
from threading import Thread, Lock
import torch

# Audio recording constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1600  # 0.1 seconds of audio (1600 samples at 16kHz)

# Load the Whisper model (using "base" for lower latency)
print("Loading Whisper model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)
print("Whisper model loaded successfully.")

# Shared audio buffer and lock for thread safety
buffer = []
buffer_lock = Lock()

# Window size: 2 seconds of audio (20 chunks of 0.1s each)
WINDOW_SIZE = 20

def get_new_text(previous_text, current_text):
    """Extract the new portion of the transcription."""
    i = 0
    min_len = min(len(previous_text), len(current_text))
    while i < min_len and previous_text[i] == current_text[i]:
        i += 1
    return current_text[i:]

def recording_thread():
    """Thread to continuously record audio and update the buffer."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    print("Recording started. Speak into the microphone...")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        with buffer_lock:
            buffer.append(data)
            if len(buffer) > WINDOW_SIZE:
                buffer.pop(0)

def transcription_thread():
    """Thread to transcribe audio every 0.5 seconds."""
    previous_text = ""
    while True:
        time.sleep(0.5)  # Process every 0.5 seconds
        # Copy the current buffer safely
        with buffer_lock:
            if not buffer:
                continue
            audio_buffer = b''.join(buffer)
        
        # Prepare audio for transcription
        start_time = time.time()
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe using faster-whisper
        segments, _ = model.transcribe(audio_np, language="en", task="transcribe")
        current_text = " ".join([seg.text for seg in segments])
        
        # Calculate latency
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract and display new transcription
        new_text = get_new_text(previous_text, current_text)
        if new_text:
            print(f"New transcription: {new_text}")
        print(f"Transcription latency: {latency:.3f} seconds")
        
        # Update previous text for the next iteration
        previous_text = current_text

def main():
    """Start the recording and transcription threads."""
    # Recording thread
    rec_thread = Thread(target=recording_thread)
    rec_thread.daemon = True
    rec_thread.start()

    # Transcription thread
    trans_thread = Thread(target=transcription_thread)
    trans_thread.daemon = True
    trans_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()