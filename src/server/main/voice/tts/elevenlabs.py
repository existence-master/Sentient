import os
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
import logging
from typing import AsyncGenerator, Optional

from .base import BaseTTS, TTSOptionsBase
from main.config import ELEVENLABS_API_KEY 

logger = logging.getLogger(__name__)

class ElevenLabsTTS(BaseTTS):
    def __init__(self, voice_id: Optional[str] = None, model_id: Optional[str] = None):
        if not ELEVENLABS_API_KEY:
            logger.error("ELEVENLABS_API_KEY environment variable not set.")
            raise ValueError("ELEVENLABS_API_KEY environment variable not set for ElevenLabsTTS.")
        
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        # Default voice "Rachel" ID: 21m00Tcm4TlvDq8ikWAM
        self.voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM" 
        self.model_id = model_id or "eleven_multilingual_v2" 
        logger.info(f"ElevenLabs TTS initialized with voice_id: {self.voice_id}, model_id: {self.model_id}")

    async def stream_tts(self, text: str, language: Optional[str] = "en", options: TTSOptionsBase = None) -> AsyncGenerator[bytes, None]:
        # The multilingual model auto-detects language from text, so the `language` param is not directly used in the API call.
        # It's kept for interface consistency.
        voice_id_to_use = self.voice_id
        model_id_to_use = self.model_id
        
        custom_settings_dict = {}
        if options:
            voice_id_to_use = options.get("voice_id", self.voice_id)
            model_id_to_use = options.get("model_id", self.model_id)
            if "stability" in options: custom_settings_dict["stability"] = options["stability"]
            if "similarity_boost" in options: custom_settings_dict["similarity_boost"] = options["similarity_boost"]
            if "style" in options: custom_settings_dict["style"] = options["style"]
            if "use_speaker_boost" in options: custom_settings_dict["use_speaker_boost"] = options["use_speaker_boost"]

        effective_settings = VoiceSettings(
            stability=custom_settings_dict.get("stability", 0.71), 
            similarity_boost=custom_settings_dict.get("similarity_boost", 0.5),
            style=custom_settings_dict.get("style", 0.0), 
            use_speaker_boost=custom_settings_dict.get("use_speaker_boost", True)
        )

        logger.info(f"ElevenLabs streaming TTS for text with detected language '{language}': '{text[:50]}...' using voice {voice_id_to_use}, model {model_id_to_use}")

        try:
            # CORRECTED: Use the client.text_to_speech.stream() method for newer library versions
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=voice_id_to_use,
                voice_settings=effective_settings,
                model_id=model_id_to_use,
                output_format="pcm_16000", 
            )

            if not hasattr(audio_stream, '__iter__') and not hasattr(audio_stream, '__aiter__'):
                logger.error("ElevenLabs client.text_to_speech.stream did not return a streamable object.")
                return

            chunk_count = 0
            for chunk in audio_stream: # type: ignore
                if chunk:
                    chunk_count += 1
                    yield chunk 
            logger.info(f"Finished streaming {chunk_count} audio chunks from ElevenLabs TTS for text: '{text[:50]}...'")
        except Exception as e:
            logger.error(f"Error during ElevenLabs TTS streaming: {e}", exc_info=True)
            yield b""
            return