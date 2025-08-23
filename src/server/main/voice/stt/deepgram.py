import logging
from typing import Tuple, Optional
from .base import BaseSTT
from main.config import DEEPGRAM_API_KEY

try:
    from deepgram import (
        DeepgramClient,
        PrerecordedOptions,
        BufferSource,
    )
except ImportError:
    raise ImportError("Deepgram SDK not installed. Please install it with 'pip install deepgram-sdk>=3.0'")

logger = logging.getLogger(__name__)

class DeepgramSTT(BaseSTT):
    """
    Deepgram Speech-to-Text implementation using Nova-3 model.
    """
    def __init__(self):
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY is required for DeepgramSTT.")
        
        try:
            self.client = DeepgramClient(DEEPGRAM_API_KEY)
            logger.info("DeepgramSTT initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Deepgram client: {e}", exc_info=True)
            raise

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> Tuple[str, Optional[str]]:
        logger.info(f"Transcribing {len(audio_bytes)} bytes of audio with Deepgram.")
        if not self.client:
            logger.error("Deepgram client not initialized.")
            return "", None

        try:
            source: BufferSource = {"buffer": audio_bytes}
            
            options = PrerecordedOptions(
                model="nova-3",
                smart_format=True,
                detect_language=True,
                punctuate=True,
                utterances=True,
                encoding="linear16",      
                sample_rate=sample_rate, 
                channels=1
            )

            response = await self.client.listen.asyncrest.v("1").transcribe_file(
                source, options
            )
            
            if response.results and response.results.channels:
                channel = response.results.channels[0]
                if channel.alternatives:
                    transcript = channel.alternatives[0].transcript
                    detected_language = channel.detected_language
                    logger.info(f"Deepgram transcription successful. Language: {detected_language}, Transcript: '{transcript[:50]}...'")
                    return transcript.strip(), detected_language
            
            logger.warning("Deepgram STT response was empty or malformed.")
            return "", None

        except Exception as e:
            logger.error(f"Error during Deepgram STT transcription: {e}", exc_info=True)
            return "", None