import pyaudio
import webrtcvad
import numpy as np
import torch
import requests
import json
import time
from csm.generator import load_csm_1b
from faster_whisper import WhisperModel
from queue import Queue
from threading import Thread, Lock

# Constants
APP_SERVER_URL = "http://localhost:5000/chat"
CHAT_ID = "voice_chat_dummy"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 480  # 30 ms at 16000 Hz
BUFFER_SECONDS = 2  # Rolling buffer size in seconds
TRANSCRIBE_INTERVAL = 0.5  # Transcribe every 0.5 seconds
SILENCE_THRESHOLD = 0.5  # 0.5 seconds of silence to detect end of speech

# Calculate derived constants
CHUNK_SIZE_SECONDS = CHUNK / RATE  # ~0.03 seconds per chunk
BUFFER_CHUNKS = int(BUFFER_SECONDS / CHUNK_SIZE_SECONDS)  # ~66 chunks for 2 seconds
SILENCE_CHUNKS = int(SILENCE_THRESHOLD / CHUNK_SIZE_SECONDS)  # ~16 chunks for 0.5 seconds

print("Script started, initializing constants...", flush=True)

# Initialize VAD
print("Initializing VAD...", flush=True)
vad = webrtcvad.Vad(3)  # Sensitivity level 3 (most sensitive), adjust if needed

# Load Whisper model
print("Loading Whisper model...", flush=True)
try:
    device = "cpu"
    compute_type = "int8"
    model = WhisperModel("base", device=device, compute_type=compute_type)
    print("Whisper model loaded successfully.", flush=True)
except Exception as e:
    print(f"Error loading Whisper model: {e}", flush=True)
    raise

# Load CSM model
print("Loading CSM model...", flush=True)
try:
    generator = load_csm_1b(device="cuda")
    print("CSM model loaded successfully.", flush=True)
except Exception as e:
    print(f"Error loading CSM model: {e}", flush=True)
    raise

# Shared variables
audio_buffer = []
buffer_lock = Lock()
full_transcription = ""
transcription_lock = Lock()
stop_speaking_queue = Queue()

def get_new_text(prev_text, new_text):
    """Extract new text by comparing previous and current transcriptions."""
    if not prev_text:
        return new_text
    prev_words = prev_text.split()
    new_words = new_text.split()
    # Find the point where new_text diverges from prev_text
    common_length = 0
    for i in range(min(len(prev_words), len(new_words))):
        if prev_words[i] != new_words[i]:
            break
        common_length += 1
    new_part = " ".join(new_words[common_length:])
    return new_part

def recording_thread():
    """Continuously record audio and detect when the user stops speaking."""
    global full_transcription  # Declare global at the start
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    is_speaking = False
    silence_counter = 0
    print("Recording started. Speak into the microphone!", flush=True)
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        with buffer_lock:
            audio_buffer.append(data)
            if len(audio_buffer) > BUFFER_CHUNKS:
                audio_buffer.pop(0)
        # VAD check
        if vad.is_speech(data, RATE):
            is_speaking = True
            silence_counter = 0
        elif is_speaking:
            silence_counter += 1
            if silence_counter >= SILENCE_CHUNKS:
                # User has stopped speaking
                with transcription_lock:
                    if full_transcription.strip():
                        stop_speaking_queue.put(full_transcription.strip())
                    full_transcription = ""  # Reset after sending
                is_speaking = False
                silence_counter = 0
                with buffer_lock:
                    audio_buffer.clear()

def transcription_thread():
    """Transcribe audio buffer periodically and accumulate transcription."""
    global full_transcription  # Declare global at the start
    prev_transcription = ""
    while True:
        time.sleep(TRANSCRIBE_INTERVAL)
        with buffer_lock:
            if not audio_buffer:
                continue
            audio_data = b''.join(audio_buffer)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        start_time = time.time()
        segments, _ = model.transcribe(audio_np, language="en", task="transcribe")
        current_transcription = " ".join([seg.text for seg in segments]).strip()
        latency = time.time() - start_time
        print(f"Transcription latency: {latency:.3f} seconds", flush=True)
        if current_transcription:
            new_text = get_new_text(prev_transcription, current_transcription)
            if new_text:
                print(f"New transcription: {new_text}", flush=True)
                with transcription_lock:
                    full_transcription = full_transcription + " " + new_text if full_transcription else new_text
            prev_transcription = current_transcription

def get_ai_response(text):
    """Get AI response from the server."""
    message = {
        "input": text,
        "pricing": "pro",
        "credits": 100,
        "chat_id": CHAT_ID
    }
    response = requests.post(APP_SERVER_URL, json=message, stream=True)
    if response.status_code != 200:
        print(f"Error from server: {response.status_code}", flush=True)
        return "Sorry, I couldn't process that."
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if data["type"] == "assistantStream":
                full_response += data["token"]
    return full_response

def synthesize_speech(text):
    """Synthesize speech from text using the CSM model."""
    audio = generator.generate(
        text=text,
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
    )
    return audio

def play_audio(audio, sample_rate):
    """Play the synthesized audio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    audio_np = audio.cpu().numpy().astype(np.float32)
    stream.write(audio_np.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    """Main function to coordinate threads and handle AI responses."""
    # Start recording thread
    rec_thread = Thread(target=recording_thread)
    rec_thread.daemon = True
    rec_thread.start()

    # Start transcription thread
    trans_thread = Thread(target=transcription_thread)
    trans_thread.daemon = True
    trans_thread.start()

    print("Voice conversation started. Speak into your microphone!", flush=True)
    while True:
        text = stop_speaking_queue.get()  # Blocks until text is available
        if text:
            print(f"You said: {text}", flush=True)
            response_text = get_ai_response(text)
            print(f"AI says: {response_text}", flush=True)
            print("Generating speech...", flush=True)
            audio_output = synthesize_speech(response_text)
            print("Speaking...", flush=True)
            play_audio(audio_output, generator.sample_rate)
            print("Listening again...", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    except Exception as e:
        print(f"Script crashed with error: {e}", flush=True)