import logging
import wave
import io
import asyncio
from typing import Tuple, Optional
from .base import BaseSTT
from main.config import ELEVENLABS_API_KEY
from elevenlabs.client import ElevenLabs

logger = logging.getLogger(__name__)

def pcm_to_wav(pcm_data: bytes, sample_rate: int, num_channels: int = 1, sample_width: int = 2) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_io.getvalue()

class ElevenLabsSTT(BaseSTT):
    def __init__(self):
        if not ELEVENLABS_API_KEY:
            logger.error("ELEVENLABS_API_KEY not set. ElevenLabsSTT cannot be initialized.")
            raise ValueError("ELEVENLABS_API_KEY is required for ElevenLabsSTT.")
        
        # Use the official ElevenLabs client, as shown in the working demo
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        logger.info("ElevenLabsSTT initialized using the official client.")

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        """Synchronous helper to run the blocking API call."""
        try:
            # Use the .speech_to_text.convert method from the library
            response = self.client.speech_to_text.convert(
                file=audio_bytes,
                model_id="scribe_v1", # A recommended model for speed
                language_code="eng",
                tag_audio_events=True,
            )
            transcription = response.text
            logger.debug(f"ElevenLabs STT Transcription: '{transcription}'")
            return transcription
        except Exception as e:
            logger.error(f"Error during ElevenLabs STT library call: {e}", exc_info=True)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                return f"ElevenLabs STT Error: {e.response.text}"
            return "Error in STT processing."

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> Tuple[str, Optional[str]]:
        try:
            # The API works best with a standard audio format like WAV
            wav_data = pcm_to_wav(audio_bytes, sample_rate)
            
            # Run the synchronous library call in a separate thread to avoid blocking
            loop = asyncio.get_running_loop()
            transcription = await loop.run_in_executor(
                None, self._transcribe_sync, wav_data
            )
            # ElevenLabs STT v1 doesn't explicitly return detected language, assumes English
            return transcription, "en"
        except Exception as e:
            logger.error(f"Error preparing audio for ElevenLabs STT: {e}", exc_info=True)
            return "Error preparing audio for STT.", None