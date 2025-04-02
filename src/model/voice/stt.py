import numpy as np
from faster_whisper import WhisperModel
import librosa

class FasterWhisperSTT:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initialize the Faster Whisper STT model.

        Args:
            model_size (str): Size of the Whisper model (e.g., "base", "tiny").
            device (str): Device to run the model on ("cpu" or "cuda").
            compute_type (str): Compute type (e.g., "int8", "float16").
        """
        try:
            print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
            self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.whisper_model = None

    def stt(self, audio: tuple[int, np.ndarray]) -> str:
        """
        Transcribe audio to text using the Faster Whisper model.

        Args:
            audio (tuple): Tuple of (sample_rate, audio_array) where audio_array is np.int16 or np.float32.

        Returns:
            str: Transcribed text.
        """
        if self.whisper_model is None:
            print("Error: Whisper model not loaded.")
            return ""

        sample_rate, audio_array = audio

        # --- FIX: Convert to float32 BEFORE resampling ---
        if audio_array.dtype == np.int16:
            # Normalize int16 to [-1.0, 1.0] range
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype != np.float32:
            # Ensure other types are converted to float32
            # (Normalization might be needed depending on the source type,
            # but for now, just casting is the direct fix for librosa)
            audio_array = audio_array.astype(np.float32)
        # --- END FIX ---

        # Resample to 16000 Hz if necessary (Now audio_array is guaranteed float32)
        if sample_rate != 16000:
            # Using y= explicitly matches librosa documentation better
            audio_array = librosa.resample(y=audio_array, orig_sr=sample_rate, target_sr=16000)

        # Ensure audio is 1D (This might be redundant if librosa.resample always returns 1D, but safe to keep)
        if audio_array.ndim != 1:
            audio_array = audio_array.flatten()

        try:
            # print(f"DEBUG: Transcribing audio with shape {audio_array.shape} and dtype {audio_array.dtype}") # Optional debug print
            segments, _ = self.whisper_model.transcribe(
                audio_array,
                language="en",
                task="transcribe"
            )
            transcription = " ".join([seg.text for seg in segments]).strip()
            # print(f"DEBUG: Transcription: {transcription}") # Optional debug print
        except Exception as e:
            print(f"Error during transcription: {e}")
            transcription = ""

        return transcription
