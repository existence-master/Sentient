import pyaudio
import webrtcvad
import numpy as np
import torch
import requests
import json
import time
from csm.generator import load_csm_1b
import speech_recognition as sr
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

# Flag to choose transcription method
USE_WHISPER = False

# Constants
APP_SERVER_URL = "http://localhost:5000/chat"
CHAT_ID = "voice_chat_dummy"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 480  # 30 ms at 16000 Hz

print("Script started, initializing constants...", flush=True)

# Initialize VAD
print("Initializing VAD...", flush=True)
vad = webrtcvad.Vad(3)

# Conditionally load Whisper model
if USE_WHISPER:
    print("Loading Whisper model...", flush=True)
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype=torch.float16,
            device="cuda:0",
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )
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

def record_speech():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("Listening...", flush=True)
    except Exception as e:
        print(f"Error opening audio stream: {e}", flush=True)
        p.terminate()
        return None

    frames = []
    silence_counter = 0
    max_silence_chunks = int(1.5 * RATE / CHUNK)
    max_recording_chunks = int(10 * RATE / CHUNK)
    chunk_count = 0
    speech_detected = False
    expected_bytes = CHUNK * 2

    while chunk_count < max_recording_chunks:
        data = stream.read(CHUNK, exception_on_overflow=False)
        if len(data) != expected_bytes:
            print(f"Warning: Incomplete chunk, got {len(data)} bytes, expected {expected_bytes}", flush=True)
            continue
        chunk_count += 1
        try:
            if vad.is_speech(data, RATE):
                frames.append(data)
                silence_counter = 0
                speech_detected = True
            else:
                if speech_detected:
                    silence_counter += 1
                    frames.append(data)
                    if silence_counter > max_silence_chunks:
                        break
                else:
                    silence_counter = 0
        except Exception as e:
            print(f"VAD error: {e}", flush=True)
            raise

    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_data = b''.join(frames) if speech_detected else None
    if audio_data and len(audio_data) / (RATE * 2) < 0.5:
        print("Audio too short for transcription.", flush=True)
        return None
    return audio_data

def transcribe_whisper(audio_data):
    if not USE_WHISPER:
        raise ValueError("Whisper is not enabled")
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    audio_input = {"raw": audio_np, "sampling_rate": RATE}
    outputs = pipe(audio_input, chunk_length_s=30, batch_size=24, return_timestamps=True)
    return outputs["text"]

def transcribe_google(audio_data):
    recognizer = sr.Recognizer()
    audio = sr.AudioData(audio_data, sample_rate=RATE, sample_width=2)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def transcribe_audio(audio_data):
    if USE_WHISPER:
        return transcribe_whisper(audio_data)
    else:
        return transcribe_google(audio_data)

def get_ai_response(text):
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
    audio = generator.generate(
        text=text,
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
    )
    return audio

def play_audio(audio, sample_rate):
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
    print("Entering main loop...", flush=True)
    print("Voice conversation started. Speak into your microphone!", flush=True)
    while True:
        audio_data = record_speech()
        if audio_data:
            text = transcribe_audio(audio_data)
            print(f"You said: {text}", flush=True)
            response_text = get_ai_response(text)
            print(f"AI says: {response_text}", flush=True)
            print("Generating speech...", flush=True)
            audio_output = synthesize_speech(response_text)
            print("Speaking...", flush=True)
            play_audio(audio_output, generator.sample_rate)
            print("Listening again...", flush=True)
        else:
            print("No speech detected. Listening again...", flush=True)
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Script crashed with error: {e}", flush=True)