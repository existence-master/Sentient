import os
from google.cloud import texttospeech
import logging

logger = logging.getLogger(__name__)

class GCPTTS:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, # FastRTC typically uses LINEAR16
            sample_rate_hertz=16000 # Common sample rate for voice calls
        )
        logger.info("GCP TTS initialized.")

    def stream_tts(self, text: str):
        """
        Synthesizes speech from the input text and yields audio chunks.
        This method mimics the stream_tts functionality of OrpheusTTS.
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Perform the text-to-speech request
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )

        # GCP TTS returns the full audio content. We need to chunk it for streaming.
        # FastRTC expects audio in chunks. A common chunk size for 16kHz 16-bit mono
        # audio is 320 bytes for 10ms of audio. Adjust as needed.
        chunk_size = 3200 # Example: 100ms of 16kHz 16-bit mono audio (16000 samples/sec * 2 bytes/sample * 0.1 sec = 3200 bytes)
        
        audio_content = response.audio_content
        for i in range(0, len(audio_content), chunk_size):
            yield audio_content[i:i + chunk_size]
        logger.debug(f"Streamed {len(audio_content)} bytes from GCP TTS.")

# How to authenticate GCP using Application Default Credentials (ADC):
# Development:
#   Run `gcloud auth application-default login` in your terminal. This will
#   open a browser window for you to log in with your Google account.
# Production:
#   Refer to the official Google Cloud documentation on "Set up Application Default Credentials"
#   for various production environments (e.g., service accounts, environment variables).
#   Documentation link: https://cloud.google.com/docs/authentication/production