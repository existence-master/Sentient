import os
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
import logging

logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    def __init__(self):
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set.")
        
        self.client = ElevenLabs(api_key=api_key)
        # You might want to make voice_id and model_id configurable
        self.voice_id = "JBFqnCBsd6RMkjVDRZzb"  # Example voice ID
        self.model_id = "eleven_multilingual_v2" # Example model ID
        logger.info("ElevenLabs TTS initialized.")

    def stream_tts(self, text: str):
        """
        Synthesizes speech from the input text and yields audio chunks.
        This method mimics the stream_tts functionality of OrpheusTTS.
        """
        # ElevenLabs can stream audio directly.
        # The output_format should be compatible with FastRTC.
        # FastRTC typically expects LINEAR16 (PCM) audio.
        # ElevenLabs 'mp3_44100_128' is MP3, which needs decoding.
        # For direct streaming to FastRTC, 'pcm_16000' is ideal.
        # Check ElevenLabs documentation for available PCM formats.
        
        # Assuming 'pcm_16000' is available and suitable for FastRTC
        # If not, an audio processing step (e.g., using pydub) would be needed
        # to convert MP3 to LINEAR16.
        
        audio_stream = self.client.text_to_speech.convert(
            text=text,
            voice=Voice(voice_id=self.voice_id, settings=VoiceSettings(stability=0.7, similarity_boost=0.8)),
            model=self.model_id,
            output_format="pcm_16000" # This is crucial for direct FastRTC compatibility
        )

        # The audio_stream is an iterator of bytes.
        # Yield chunks directly from the stream.
        for chunk in audio_stream:
            yield chunk
        logger.debug(f"Streamed audio from ElevenLabs TTS for text: '{text[:50]}...'")