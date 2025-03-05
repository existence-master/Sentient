import webrtcvad
from openwakeword.model import Model
import numpy as np
import pyaudio
import threading
from queue import Queue
import time
import os
import subprocess
import socket

class WakeWordDetector:
    def __init__(self, mode="development"):
        self.vad = webrtcvad.Vad(3)
        self.model = Model(wakeword_models=["Hey_Sen_she_ent.onnx"], inference_framework="onnx")
        self.sample_rate = 16000
        self.frame_duration_ms = 80
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.running = True
        self.pa_format = pyaudio.paInt16
        self.channels = 1
        self.chunk_size = self.sample_rate * 1
        self.buffer_size = int(1 * self.sample_rate)
        self.buffer = np.zeros(self.chunk_size * 2)
        self.mic_queue = Queue()
        self.pa = pyaudio.PyAudio()
        self.mode = mode
        if self.mode == "development":
            self.app_path = r"D:\Documents\cyber\projects\Sentient-New\Code\src\interface"
        else:
            raise ValueError("Only development mode is supported for now.")
        self.frontend_process = None
        self.dev_process = None

    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        if self.running:
            audio_frame = np.frombuffer(in_data, dtype=np.int16)
            self.mic_queue.put(audio_frame)
        return (None, pyaudio.paContinue)

    def _mic_thread(self):
        try:
            stream = self.pa.open(
                format=self.pa_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback,
                start=True
            )
            while self.running:
                time.sleep(0.01)
            stream.stop_stream()
            stream.close()
        except OSError as e:
            print(f"Microphone error: {e}")

    def _process_audio(self):
        while self.running:
            try:
                raw_audio = self.mic_queue.get(timeout=0.1)
                self.buffer = np.roll(self.buffer, -len(raw_audio))
                self.buffer[-len(raw_audio):] = raw_audio
                if len(raw_audio) == self.chunk_size:
                    prediction = self.model.predict(self.buffer)
                    print(f"Confidence: {prediction['Hey_Sen_she_ent']:.4f}")
                    if prediction["Hey_Sen_she_ent"] > 0.0080:
                        print("\n=== WAKE WORD DETECTED ===")
                        self._trigger_action()
                        time.sleep(2)
            except Exception as e:
                if str(e):
                    print(f"Processing error: {e}")

    def _trigger_action(self):
        print("Launching main application...")
        self.launch_app_and_server()

    def launch_app_and_server(self):
        try:
            if self.is_app_running():
                print("App already running, signaling to start voice conversation...")
                self.send_signal_to_app()
            else:
                print("Launching development app with npm run dev...")
                creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                self.dev_process = subprocess.Popen(
                    ["npm", "run", "dev", "--", "--wake-word"],
                    cwd=self.app_path,
                    creationflags=creationflags
                )
                time.sleep(5)  # Wait for app to start
                if self.is_app_running():
                    print("Development app launched successfully.")
                else:
                    print("Failed to launch development app.")
        except Exception as e:
            print(f"Error launching app: {str(e)}")

    def is_app_running(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("localhost", 2345))
                return True
        except (socket.timeout, ConnectionRefusedError):
            return False
        except Exception as e:
            print(f"Error checking app status: {e}")
            return False

    def send_signal_to_app(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("localhost", 2345))
                s.sendall("start-voice-conversation".encode())
        except Exception as e:
            print(f"Error sending signal: {e}")

    def start(self):
        mic_thread = threading.Thread(target=self._mic_thread)
        proc_thread = threading.Thread(target=self._process_audio)
        mic_thread.start()
        proc_thread.start()
        mic_thread.join()
        proc_thread.join()

    def stop(self):
        self.running = False
        time.sleep(0.5)
        if self.dev_process:
            self.dev_process.terminate()
        self.pa.terminate()

if __name__ == "__main__":
    detector = WakeWordDetector()
    print("Listening for 'Hey Sentient'...")
    detector.start()