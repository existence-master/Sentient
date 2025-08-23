import asyncio
import logging
import threading
from typing import AsyncGenerator, Tuple, Optional
import numpy as np
from smallestai.waves import WavesStreamingTTS, TTSConfig
from main.config import SMALLEST_AI_API_KEY
from .base import BaseTTS, TTSOptionsBase

logger = logging.getLogger(__name__)

class SmallestAITTS(BaseTTS):
    """
    Text-to-Speech implementation using the Smallest AI API.
    """
    def __init__(self):
        if not SMALLEST_AI_API_KEY:
            raise ValueError("SMALLEST_AI_API_KEY is not set.")
        logger.info("SmallestAITTS initialized.")

    async def stream_tts(self, text: str, language: Optional[str] = "en", options: TTSOptionsBase = None) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        opts = options or {}
        
        lang_short = language.split('-')[0] if language else 'en'

        # Smallest AI has specific voices for languages.
        # From their docs, 'nyah' for 'en', 'lakshya' for 'hi'.
        # We'll default to 'nyah' for other languages as a fallback.
        voice_id_map = {
            "en": "nyah",
            "hi": "lakshya"
        }
        voice_id = voice_id_map.get(lang_short, "nyah")
        
        # According to docs, lightning-large supports consistency/similarity
        model = "lightning-large"

        config = TTSConfig(
            api_key=SMALLEST_AI_API_KEY,
            voice_id=voice_id,
            language=lang_short,
            sample_rate=opts.get("sample_rate", 24000),
            speed=opts.get("speed", 1.0),
            consistency=opts.get("consistency", 0.5),
            similarity=opts.get("similarity", 0.0),
            enhancement=opts.get("enhancement", 1),
        )
        logger.info(f"Starting SmallestAI TTS stream for text: '{text[:50]}...' with config: {config}")
        
        tts_streamer = WavesStreamingTTS(config)

        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def producer():
            try:
                for chunk in tts_streamer.synthesize(text):
                    loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
            except Exception as e:
                logger.error(f"Error in SmallestAI TTS producer thread: {e}", exc_info=True)
                loop.call_soon_threadsafe(audio_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        chunk_count = 0
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            
            chunk_count += 1
            # Smallest AI returns raw PCM bytes, 16-bit signed little-endian.
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            yield (config.sample_rate, audio_array)
        
        producer_thread.join()        logger.info(f"Finished streaming {chunk_count} audio chunks from SmallestAI TTS.")
