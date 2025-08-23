import snac
import torch
import numpy as np
import asyncio
from llama_cpp import Llama, CreateCompletionStreamResponse
import os
import threading
import traceback
import logging
from typing import (
    AsyncGenerator,
    Generator,
    Literal,
    Optional,
    TypedDict,
    cast,
    Iterator,
    Tuple
)
from numpy.typing import NDArray

from .base import BaseTTS, TTSOptionsBase

logger = logging.getLogger(__name__)

AVAILABLE_VOICES_ORPHEUS: list[Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]] = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE_ORPHEUS: Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"] = "tara"

class OrpheusTTSOptions(TTSOptionsBase):
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    voice_id: Optional[Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]] = None

SAMPLE_RATE_ORPHEUS = 24000
CUSTOM_TOKEN_PREFIX_ORPHEUS = "<custom_token_"

class OrpheusTTS(BaseTTS):
    SAMPLE_RATE = SAMPLE_RATE_ORPHEUS
    AVAILABLE_VOICES = AVAILABLE_VOICES_ORPHEUS
    DEFAULT_VOICE = DEFAULT_VOICE_ORPHEUS
    CUSTOM_TOKEN_PREFIX = CUSTOM_TOKEN_PREFIX_ORPHEUS

    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int | None = None,
        verbose: bool = True,
        n_ctx: int = 2048,
        default_voice_id: Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"] = DEFAULT_VOICE_ORPHEUS
    ):
        from main.config import ORPHEUS_MODEL_PATH as ORPHEUS_MODEL_PATH
        from main.config import ORPHEUS_N_GPU_LAYERS as MAIN_ORPHEUS_N_GPU_LAYERS

        self.model_path = model_path or ORPHEUS_MODEL_PATH
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else MAIN_ORPHEUS_N_GPU_LAYERS
        self.verbose = verbose
        self.n_ctx = n_ctx

        self.instance_default_voice = default_voice_id if default_voice_id in self.AVAILABLE_VOICES else self.DEFAULT_VOICE

        self.snac_model = None
        self.llm = None
        self.snac_device = None
        self._load_models()

    def _load_models(self):
        if not self.snac_model:
            try:
                self.snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval() # type: ignore
                self.snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                self.snac_model.to(self.snac_device)
                logger.info(f"SNAC model loaded on device: {self.snac_device}")
            except Exception as e:
                logger.error(f"Error loading SNAC model for OrpheusTTS: {e}", exc_info=True)
                raise

        if not self.llm:
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Orpheus model not found at path: {self.model_path}")
            try:
                self.llm = Llama(
                    model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose, n_ctx=self.n_ctx, logits_all=True
                )
                logger.info("Orpheus model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Orpheus GGUF model: {e}", exc_info=True)
                raise

    def _format_prompt(self, prompt: str, voice: Optional[str] = None) -> str:
        resolved_voice = voice or self.instance_default_voice
        return f"<|audio|>{resolved_voice}: {prompt}<|eot_id|>"

    def _turn_token_into_id(self, token_string: str, index: int) -> int | None:
        last_token_start = token_string.rfind(self.CUSTOM_TOKEN_PREFIX)
        if last_token_start == -1: return None
        last_token = token_string[last_token_start:]
        if last_token.startswith(self.CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[len(self.CUSTOM_TOKEN_PREFIX):-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except (ValueError, IndexError): return None
        return None

    def _snac_decode_sync(self, multiframe: list[int]) -> NDArray[np.float32] | None:
        if not self.snac_model or len(multiframe) < 28: return None
        frame = multiframe[-28:]
        try:
            codes_0 = torch.tensor([frame[i] for i in range(0, 28, 7)], device=self.snac_device, dtype=torch.int32).unsqueeze(0)
            codes_1 = torch.tensor([frame[i] for i in [1, 4, 8, 11, 15, 18, 22, 25]], device=self.snac_device, dtype=torch.int32).unsqueeze(0)
            codes_2 = torch.tensor([frame[i] for i in [2,3,5,6,9,10,12,13,16,17,19,20,23,24,26,27]], device=self.snac_device, dtype=torch.int32).unsqueeze(0)
            with torch.inference_mode(): audio_hat = self.snac_model.decode([codes_0, codes_1, codes_2])
            return audio_hat[:, :, 2048:4096].squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"Error during SNAC decoding: {e}", exc_info=True)
            return None

    def _generate_tokens_sync(self, text: str, options: OrpheusTTSOptions) -> Generator[str, None, None]:
        if not self.llm: yield "<|error|>"; return
        voice = options.get("voice_id")
        formatted_prompt = self._format_prompt(text, voice)
        try:
            response_stream = self.llm(
                prompt=formatted_prompt, max_tokens=options.get("max_tokens", 4096),
                temperature=options.get("temperature", 0.7), top_p=options.get("top_p", 0.9),
                repeat_penalty=options.get("repetition_penalty", 1.1), stream=True,
                stop=["<|eot_id|>", "<|endoftext|>"]
            )
            for data in cast(Iterator[CreateCompletionStreamResponse], response_stream):
                yield data['choices'][0].get('text', '')
        except Exception as e:
            logger.error(f"Error generating Orpheus tokens: {e}", exc_info=True); yield "<|error|>"

    async def stream_tts(self, text: str, language: Optional[str] = "en", options: TTSOptionsBase = None) -> AsyncGenerator[Tuple[int, NDArray[np.float32]], None]:
        # Orpheus is English-only, so we ignore the language parameter but keep it for interface consistency.
        if language and not language.startswith("en"):
            logger.warning(f"OrpheusTTS received a non-English language code '{language}'. It will attempt to synthesize the text as English.")
        
        logger.info(f"Starting Orpheus TTS stream for text: '{text[:50]}...'")
        opts = cast(OrpheusTTSOptions, options or {})
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def thread_worker():
            try:
                buffer, count = [], 0
                for token_text in self._generate_tokens_sync(text, opts):
                    if "<|error|>" in token_text: break
                    token_id = self._turn_token_into_id(token_text, count)
                    if token_id is not None and token_id >= 0:
                        buffer.append(token_id); count += 1
                        if count % 7 == 0 and len(buffer) >= 28:
                            audio_slice = self._snac_decode_sync(buffer)
                            if audio_slice is not None and audio_slice.size > 0:
                                loop.call_soon_threadsafe(queue.put_nowait, (self.SAMPLE_RATE, audio_slice))
            except Exception as e:
                logger.error(f"Error in OrpheusTTS worker thread: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=thread_worker, daemon=True)
        thread.start()
        chunk_count = 0
        while True:
            chunk = await queue.get()
            if chunk is None: break
            chunk_count += 1
            yield chunk
        await asyncio.to_thread(thread.join)        
        logger.info(f"Finished streaming {chunk_count} audio chunks from Orpheus TTS.")
