import logging
from typing import Tuple, Optional
from .base import BaseSTT
from main.config import DEEPGRAM_API_KEY

from deepgram import AsyncDeepgramClient


logger = logging.getLogger(__name__)


class DeepgramSTT(BaseSTT):
    """
    Deepgram Speech-to-Text implementation using Nova-3 model.

    Uses deepgram-sdk v6+ (``AsyncDeepgramClient.listen.v1.media``); the legacy
    ``ListenRESTOptions`` / ``listen.asyncrest`` API was removed in favor of
    Fern-generated clients.
    """
    def __init__(self):
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY is required for DeepgramSTT.")

        try:
            self.client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
            self._media = self.client.listen.v1.media
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
            # Raw PCM requires sample_rate on the query string; v6 media client does not
            # expose it as a typed arg, so pass via request_options.
            response = await self._media.transcribe_file(
                request=audio_bytes,
                model="nova-3",
                smart_format=True,
                detect_language=True,
                punctuate=True,
                utterances=True,
                encoding="linear16",
                request_options={
                    "additional_query_parameters": {"sample_rate": sample_rate},
                },
            )

            if getattr(response, "results", None) and response.results.channels:
                channel = response.results.channels[0]
                if channel.alternatives:
                    transcript = channel.alternatives[0].transcript
                    detected_language = channel.detected_language
                    preview = (transcript or "")[:50]
                    logger.info(
                        f"Deepgram transcription successful. Language: {detected_language}, "
                        f"Transcript: '{preview}...'"
                    )
                    return (transcript or "").strip(), detected_language

            logger.warning("Deepgram STT response was empty or malformed.")
            return "", None

        except Exception as e:
            logger.error(f"Error during Deepgram STT transcription: {e}", exc_info=True)
            return "", None
