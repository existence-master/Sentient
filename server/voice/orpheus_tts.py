# src/model/voice/orpheus_tts.py
import snac
import torch
import numpy as np
import asyncio
from llama_cpp import Llama, CreateCompletionStreamResponse
import os
from dotenv import load_dotenv
import threading
import time
from typing import (
    AsyncGenerator,
    Generator,
    Literal,
    Optional,
    TypedDict,
    cast,
    Iterator
)
from numpy.typing import NDArray

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Define TTSOptions ---
# Define the Literal type for voice IDs
VoiceId = Literal["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

class TTSOptions(TypedDict, total=False):
    """Options for Orpheus TTS generation."""
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    voice_id: VoiceId # Use the defined Literal type

# --- Constants ---
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "orpheus-3b-0.1-ft-q4_k_m.gguf")
DEFAULT_N_GPU_LAYERS = int(os.getenv("ORPHEUS_N_GPU_LAYERS", 30))
SAMPLE_RATE = 24000
AVAILABLE_VOICES: list[VoiceId] = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE: VoiceId = "tara" # Ensure DEFAULT_VOICE is typed
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# --- Orpheus TTS Class implementing the Protocol ---
class OrpheusTTS:
    SAMPLE_RATE = SAMPLE_RATE
    AVAILABLE_VOICES = AVAILABLE_VOICES
    DEFAULT_VOICE = DEFAULT_VOICE # Class default
    CUSTOM_TOKEN_PREFIX = CUSTOM_TOKEN_PREFIX

    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int | None = None,
        verbose: bool = False,
        n_ctx: int = 4096,
        default_voice_id: VoiceId = DEFAULT_VOICE # Add parameter here
    ):
        """
        Initializes the OrpheusTTS model, loading required components.

        Args:
            model_path: Path to the Orpheus GGUF model file. Uses env/default if None.
            n_gpu_layers: Number of layers to offload to GPU. Uses env/default if None.
            verbose: Enable verbose logging from llama_cpp.
            n_ctx: Context size for llama_cpp. Defaults to 2048. Avoid 0.
            default_voice_id: The default voice to use for this instance if not specified in options.
        """
        self.model_path = model_path or os.getenv("ORPHEUS_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else DEFAULT_N_GPU_LAYERS
        self.verbose = verbose
        self.n_ctx = n_ctx

        # --- Set and validate the instance's default voice ---
        if default_voice_id in self.AVAILABLE_VOICES:
            self.instance_default_voice: VoiceId = default_voice_id
        else:
            print(f"Warning: Provided default_voice_id '{default_voice_id}' is invalid. "
                  f"Falling back to class default '{self.DEFAULT_VOICE}'.")
            self.instance_default_voice = self.DEFAULT_VOICE
        print(f"OrpheusTTS instance configured with default voice: {self.instance_default_voice}")
        # --- End voice validation ---

        self.snac_model = None
        self.llm = None
        self.snac_device = "cpu"
        self._load_models()

    # _load_models remains the same...
    def _load_models(self):
        """Loads the SNAC vocoder and Orpheus LLM."""
        # Load SNAC Model
        if self.snac_model is None:
            try:
                print("Loading SNAC model...")
                # Ensure weights_only=True for security unless strictly needed otherwise
                self.snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz", weights_only=True).eval()
                self.snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                print(f"SNAC model using device: {self.snac_device}")
                self.snac_model = self.snac_model.to(self.snac_device)
                print("SNAC model loaded.")
            except TypeError: # Fallback for older snac versions without weights_only
                 print("Loading SNAC model (without weights_only)...")
                 self.snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
                 self.snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                 print(f"SNAC model using device: {self.snac_device}")
                 self.snac_model = self.snac_model.to(self.snac_device)
                 print("SNAC model loaded.")
            except Exception as e:
                print(f"Error loading SNAC model: {e}")
                raise RuntimeError("Failed to load SNAC model") from e

        # Load Orpheus Model (llama_cpp)
        if self.llm is None:
            if not os.path.exists(self.model_path):
                print(f"Error: Orpheus model not found at {self.model_path}")
                raise FileNotFoundError(f"Orpheus model not found at {self.model_path}")

            try:
                print(f"Loading Orpheus model from {self.model_path}...")
                # **FIX: Use specified n_ctx, default to 2048, DO NOT USE 0**
                print(f"Using n_ctx={self.n_ctx}")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                    n_ctx=self.n_ctx # Explicitly set context size
                )
                print(f"Orpheus model loaded with {self.n_gpu_layers} layers offloaded to GPU.")
            except Exception as e:
                print(f"Error loading Orpheus model: {e}")
                raise RuntimeError("Failed to load Orpheus LLM") from e


    # _format_prompt: Use the instance default voice if voice argument is None
    def _format_prompt(self, prompt: str, voice: VoiceId | None = None) -> str:
        """Format the prompt using the provided voice or the instance default."""
        # Use provided voice, else instance default, else class default (shouldn't happen due to __init__)
        resolved_voice = voice or self.instance_default_voice

        # This warning should now only trigger if an *invalid* voice is passed via options
        if voice is not None and voice not in self.AVAILABLE_VOICES:
            print(f"Warning: Voice '{voice}' specified in options is invalid. Using instance default '{self.instance_default_voice}' instead.")
            resolved_voice = self.instance_default_voice

        formatted_prompt = f"{resolved_voice}: {prompt}"
        special_start = "<|audio|>"
        special_end = "<|eot_id|>"
        return f"{special_start}{formatted_prompt}{special_end}"

    # _turn_token_into_id remains the same...
    def _turn_token_into_id(self, token_string: str, index: int) -> int | None:
        """Convert token string using the formula from the working test script."""
        token_string = token_string.strip()
        last_token_start = token_string.rfind(self.CUSTOM_TOKEN_PREFIX)
        if last_token_start == -1:
            return None # No custom token prefix found

        last_token = token_string[last_token_start:]
        # print(f"DEBUG: Processing token: '{last_token}' at index {index}") # Optional Debug
        if last_token.startswith(self.CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[len(self.CUSTOM_TOKEN_PREFIX):-1]
                # **FIX: Use the exact formula from the working test script**
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                # print(f"DEBUG: Extracted ID: {token_id}") # Optional Debug
                return token_id
            except ValueError:
                # print(f"DEBUG: Failed to parse number from '{last_token}'") # Optional Debug
                return None
        return None # Token doesn't match expected format

    # _snac_decode_sync remains the same...
    def _snac_decode_sync(self, multiframe: list[int]) -> NDArray[np.float32] | None:
        """Convert token frames to audio using SNAC, filtering invalid IDs."""
        if self.snac_model is None:
            print("Error: SNAC model not loaded in _snac_decode_sync")
            return None

        # --- Filter for Valid Token IDs ---
        # Assume a common codebook size leading to range [0, 4095]
        # Adjust VALID_MAX_ID if you know the exact codebook size of snac_24khz
        VALID_MIN_ID = 0
        VALID_MAX_ID = 4095 # Assuming 4096 codebook size
        original_len = len(multiframe)

        # Keep only IDs within the valid range
        valid_ids = [id for id in multiframe if VALID_MIN_ID <= id <= VALID_MAX_ID]

        # Log if we filtered anything
        if len(valid_ids) != original_len:
            print(f"DEBUG: Filtered {original_len - len(valid_ids)} invalid token IDs. "
                  f"Original range: ({min(multiframe)}, {max(multiframe)}), "
                  f"Valid range: ({min(valid_ids) if valid_ids else 'N/A'}, {max(valid_ids) if valid_ids else 'N/A'})")

        # Check if we still have enough *valid* tokens for decoding (e.g., 28)
        if len(valid_ids) < 28:
            # print(f"DEBUG: Insufficient valid tokens ({len(valid_ids)} < 28) after filtering. Skipping frame.")
            return None
        # --- End Filtering ---

        # Use the last 28 *valid* tokens for processing
        frame = valid_ids[-28:]
        num_frames = len(frame) // 7 # Should be 4

        # Build tensors using the *filtered* 'frame' list
        codes_0 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.snac_device, dtype=torch.int32)

        for j in range(num_frames):
            i = 7*j
            # Ensure indices are valid for the filtered 'frame' list
            if i+6 < len(frame): # Basic boundary check
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])
            else:
                # This shouldn't happen if len(frame) % 7 == 0 and len(frame) >= 28
                print(f"WARNING: Indexing error during tensor building. i={i}, len(frame)={len(frame)}")
                return None # Avoid potential error

        # Check if any tensors are empty (shouldn't happen if len(frame) >= 28)
        if codes_0.numel() == 0 or codes_1.numel() == 0 or codes_2.numel() == 0:
             print("WARNING: Empty code tensors after building loop. Skipping frame.")
             return None

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

        # Sanity check (optional, should be guaranteed by filtering)
        # if (torch.any(codes[0] < VALID_MIN_ID) or torch.any(codes[0] > VALID_MAX_ID) or
        #     torch.any(codes[1] < VALID_MIN_ID) or torch.any(codes[1] > VALID_MAX_ID) or
        #     torch.any(codes[2] < VALID_MIN_ID) or torch.any(codes[2] > VALID_MAX_ID)):
        #      print(f"WARNING: Invalid token IDs detected *after* filtering. This is unexpected.")
        #      return None

        try:
            with torch.inference_mode():
                audio_hat = self.snac_model.decode(codes) # Expected shape: [B, 1, T_audio]

            audio_slice_tensor = audio_hat[:, :, 2048:4096].squeeze()
            audio_slice_np = audio_slice_tensor.cpu().numpy().astype(np.float32)

            if audio_slice_np.ndim > 1:
                audio_slice_np = audio_slice_np.flatten()

            return audio_slice_np

        # Catch potential runtime errors specifically during decode
        except RuntimeError as e:
            # Check if it's the CUDA assert error
             if "CUDA error: device-side assert triggered" in str(e):
                  print(f"ERROR: CUDA device-side assert caught during snac_model.decode. Likely due to remaining invalid indices.")
                  # Log more details about the input tensors
                  print(f"Input shapes: c0={codes[0].shape}, c1={codes[1].shape}, c2={codes[2].shape}")
                  print(f"Input ranges: c0=({codes[0].min()},{codes[0].max()}), c1=({codes[1].min()},{codes[1].max()}), c2=({codes[2].min()},{codes[2].max()})")
                  return None # Prevent crash propagation
             else:
                  print(f"RuntimeError during SNAC decoding: {e}")
                  return None # Handle other runtime errors
        except Exception as e:
            print(f"Unexpected error during SNAC decoding: {e}")
            traceback.print_exc() # Print full traceback for unexpected errors
            return None

    # _generate_tokens_sync: Use the instance default voice if not provided in options
    def _generate_tokens_sync(self, text: str, options: TTSOptions | None = None) -> Generator[str, None, None]:
        """Synchronously generate token stream from the Orpheus model."""
        if self.llm is None:
            print("Error: Orpheus LLM not loaded.")
            yield "<|error|>"
            return

        opts = options or {}
        # --- Use instance default voice as fallback ---
        voice = opts.get("voice_id", self.instance_default_voice)
        # --- End voice logic change ---
        formatted_prompt = self._format_prompt(text, voice) # _format_prompt also validates

        max_tokens = opts.get("max_tokens", 4096)
        temperature = opts.get("temperature", 0.7)
        top_p = opts.get("top_p", 0.9)
        repetition_penalty = opts.get("repetition_penalty", 1.1)

        try:
            response_stream = self.llm(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                stream=True
            )
            for data in cast(Iterator[CreateCompletionStreamResponse], response_stream):
                if 'choices' in data and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    if 'text' in choice:
                        token_text = choice['text']
                        if token_text:
                            yield token_text
        except Exception as e:
            print(f"Error during Orpheus token generation: {e}")
            yield "<|error|>"

    # stream_tts_sync, stream_tts, tts remain the same as they respect the options dict
    # which now correctly falls back to the instance default voice via _generate_tokens_sync
    def stream_tts_sync(
        self, text: str, options: TTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Synchronous streaming TTS using verified logic."""
        # The call to _generate_tokens_sync now uses the correct default voice
        token_gen = self._generate_tokens_sync(text, options)
        buffer = []
        count = 0

        for token_text in token_gen:
            if token_text == "<|error|>":
                print("Error signal received from token generator.")
                break

            token_id = self._turn_token_into_id(token_text, count)

            if token_id is not None and isinstance(token_id, int) and token_id > 0:
                buffer.append(token_id)
                count += 1

                if count % 7 == 0 and len(buffer) >= 28:
                    audio_slice_np = self._snac_decode_sync(buffer)
                    if audio_slice_np is not None and audio_slice_np.size > 0:
                        yield (self.SAMPLE_RATE, audio_slice_np)

    async def stream_tts(
        self, text: str, options: TTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Asynchronous streaming TTS wrapper."""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        finished = asyncio.Event()

        # Pass options down to the sync function
        def thread_worker():
            try:
                for chunk in self.stream_tts_sync(text, options): # Pass options here
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                print(f"Error in TTS worker thread: {e}")
                loop.call_soon_threadsafe(queue.put_nowait, None)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)
                loop.call_soon_threadsafe(finished.set)

        thread = threading.Thread(target=thread_worker, daemon=True)
        thread.start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                queue.task_done()
                break
            yield chunk
            queue.task_done()

        await finished.wait()

    def tts(
        self, text: str, options: TTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        """Synchronous TTS generating the entire audio."""
        all_chunks = []
        # Pass options down to the sync function
        for sr, chunk_np in self.stream_tts_sync(text, options): # Pass options here
            if chunk_np is not None and chunk_np.size > 0:
                 if chunk_np.ndim > 1: chunk_np = chunk_np.flatten()
                 if chunk_np.dtype != np.float32: chunk_np = chunk_np.astype(np.float32)
                 all_chunks.append(chunk_np)

        if not all_chunks:
            print("Warning: No audio chunks generated for full TTS.")
            full_audio = np.array([], dtype=np.float32)
        else:
            full_audio = np.concatenate(all_chunks)

        return (self.SAMPLE_RATE, full_audio)

# --- Example Usage (for testing this script directly) ---
# async def main_test():
#     print("Initializing OrpheusTTS with specific default voice...")
#     try:
#         # Test setting default voice during init
#         tts_model = OrpheusTTS(verbose=False, n_ctx=2048, default_voice_id="mia") # Set default to mia
#     except Exception as e:
#         print(f"Failed to initialize model: {e}")
#         return

#     test_text = "This should be spoken using the default voice set during initialization."
#     print(f"\n--- Testing Async Stream (using instance default: {tts_model.instance_default_voice}) ---")
#     start_time = time.time()
#     try:
#         # Don't pass voice_id in options, let the instance default take effect
#         audio_stream = tts_model.stream_tts(test_text, options={}) # Empty options
#         async for i, (sr, chunk) in enumerate(audio_stream):
#             if i == 0: print(f"First chunk received...")
#             await asyncio.sleep(0.01) # Minimal sleep
#     except Exception as e:
#          print(f"Error during async streaming test: {e}")
#          traceback.print_exc()
#     finally:
#         end_time = time.time()
#         print(f"Async stream finished in {end_time - start_time:.2f}s")

#     # Test overriding the default voice via options
#     test_text_override = "This sentence should use the 'jess' voice, overriding the default."
#     override_voice: VoiceId = "jess"
#     print(f"\n--- Testing Async Stream (overriding default with: {override_voice}) ---")
#     start_time = time.time()
#     try:
#         # Pass voice_id in options to override the instance default
#         audio_stream_override = tts_model.stream_tts(test_text_override, options={"voice_id": override_voice})
#         async for i, (sr, chunk) in enumerate(audio_stream_override):
#              if i == 0: print(f"First chunk received...")
#              await asyncio.sleep(0.01)
#     except Exception as e:
#          print(f"Error during async streaming test: {e}")
#          traceback.print_exc()
#     finally:
#         end_time = time.time()
#         print(f"Async stream finished in {end_time - start_time:.2f}s")


# if __name__ == "__main__":
#     asyncio.run(main_test())