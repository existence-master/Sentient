# src/model/voice/stt.py
import numpy as np
from faster_whisper import WhisperModel
import librosa # Add 'librosa' to requirements.txt
import asyncio

# --- Constants ---
WHISPER_MODEL_SIZE = "base" # Or "tiny", "small", etc.
DEVICE = "cpu" # Or "cuda" if available
COMPUTE_TYPE = "int8" # Or "float16" for GPU
EXPECTED_SAMPLE_RATE = 16000

# --- Global Model Initialization ---
try:
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})...")
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_model = None

async def transcribe_audio(audio_array: np.ndarray, sample_rate: int) -> str:
    """
    Transcribes audio using the preloaded Faster Whisper model.
    Handles resampling and type conversion.
    """
    if whisper_model is None:
        print("Error: Whisper model not loaded.")
        return ""

    start_time = asyncio.get_event_loop().time()

    # 1. Resample if necessary
    if sample_rate != EXPECTED_SAMPLE_RATE:
        # print(f"Resampling audio from {sample_rate}Hz to {EXPECTED_SAMPLE_RATE}Hz...")
        try:
            # Ensure audio is float32 for librosa
            if not np.issubdtype(audio_array.dtype, np.floating):
                 # Assuming int16 input -> normalize to [-1, 1] float32
                 audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.float64:
                 audio_array = audio_array.astype(np.float32)

            # Ensure mono
            if audio_array.ndim > 1 and audio_array.shape[1] > 1:
                 audio_array = librosa.to_mono(audio_array.T) # librosa expects channels last for to_mono

            audio_array = await asyncio.to_thread(
                librosa.resample,
                audio_array,
                orig_sr=sample_rate,
                target_sr=EXPECTED_SAMPLE_RATE
            )
            sample_rate = EXPECTED_SAMPLE_RATE
            # print("Resampling complete.")
        except Exception as e:
            print(f"Error during resampling: {e}")
            return ""

    # 2. Ensure correct dtype (float32) for Whisper
    if not np.issubdtype(audio_array.dtype, np.floating):
         # Assuming int16 input -> normalize to [-1, 1] float32
         audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.float64:
         audio_array = audio_array.astype(np.float32)

    # 3. Transcribe using asyncio.to_thread for the blocking call
    try:
        # print("Starting transcription...")
        segments, info = await asyncio.to_thread(
            whisper_model.transcribe,
            audio_array,
            language="en",
            task="transcribe"
        )
        transcription = " ".join([seg.text for seg in segments]).strip()
        # print(f"Transcription successful: '{transcription}'")
    except Exception as e:
        print(f"Error during transcription: {e}")
        transcription = ""

    end_time = asyncio.get_event_loop().time()
    # print(f"Transcription latency: {end_time - start_time:.3f} seconds")
    return transcription