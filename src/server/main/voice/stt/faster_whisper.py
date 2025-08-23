import numpy as np
from faster_whisper import WhisperModel
import librosa
import logging
import asyncio 
from typing import Tuple, Optional

from .base import BaseSTT
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FasterWhisperSTT(BaseSTT):
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        try:
            logger.info(f"Loading FasterWhisper model '{model_size}' on {device} ({compute_type})...")
            self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("FasterWhisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading FasterWhisper model: {e}")
            self.whisper_model = None
            raise 

    def _transcribe_sync(self, audio_float32: np.ndarray) -> Tuple[str, Optional[str]]:
        if self.whisper_model is None:
            logger.error("FasterWhisper model not loaded. Cannot transcribe.")
            return "", None
        segments, info = self.whisper_model.transcribe(
            audio_float32,
            language=None, # Auto-detect language
            task="transcribe"
        )
        transcription = " ".join([seg.text for seg in segments]).strip()
        detected_language = info.language
        return transcription, detected_language

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> Tuple[str, Optional[str]]:
        if self.whisper_model is None:
            return "", None

        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_np.astype(np.float32) / 32768.0
            
            target_sr = 16000 
            if sample_rate != target_sr:
                audio_float32 = librosa.resample(y=audio_float32, orig_sr=sample_rate, target_sr=target_sr)

            if audio_float32.ndim != 1: 
                audio_float32 = audio_float32.flatten()
            
            loop = asyncio.get_running_loop()
            transcription, detected_language = await loop.run_in_executor(None, self._transcribe_sync, audio_float32)
            
            logger.info(f"FasterWhisper Transcription: '{transcription}', Language: {detected_language}")
            return transcription, detected_language
        except Exception as e:
            logger.error(f"Error during FasterWhisper STT transcription: {e}", exc_info=True)
            return "", None