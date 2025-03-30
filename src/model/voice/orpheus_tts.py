# src/model/voice/orpheus_tts.py
import snac
import torch
import numpy as np
import asyncio
from llama_cpp import Llama
import os
from dotenv import load_dotenv

# Load environment variables specifically for model path if needed
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "orpheus-3b-0.1-ft-q4_k_m.gguf")
# Ensure N_GPU_LAYERS is read from env or set appropriately
N_GPU_LAYERS = int(os.getenv("ORPHEUS_N_GPU_LAYERS", 30))
SAMPLE_RATE = 24000
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# --- Global Model Initialization ---
snac_model = None
llm = None
snac_device = "cpu"

def load_models():
    global snac_model, llm, snac_device
    if snac_model is None:
        try:
            print("Loading SNAC model...")
            snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"SNAC model using device: {snac_device}")
            snac_model = snac_model.to(snac_device)
            print("SNAC model loaded.")
        except Exception as e:
            print(f"Error loading SNAC model: {e}")

    if llm is None:
        if not os.path.exists(MODEL_PATH):
             print(f"Error: Orpheus model not found at {MODEL_PATH}")
             return # Prevent crash if model is missing

        try:
            print("Loading Orpheus model...")
            # Consider adding n_ctx if needed, e.g., n_ctx=2048
            llm = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS, verbose=False) # Set verbose=False for cleaner logs
            print(f"Orpheus model loaded with {N_GPU_LAYERS} layers offloaded to GPU.")
        except Exception as e:
            print(f"Error loading Orpheus model: {e}")

# Call load_models() on import? Or explicitly call from app.py startup
# load_models() # Let's call it from app.py startup for better control

# --- Helper Functions (mostly unchanged) ---
def format_prompt(prompt, voice=DEFAULT_VOICE):
    """Format the prompt with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
    formatted_prompt = f"{voice}: {prompt}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    return f"{special_start}{formatted_prompt}{special_end}"

def turn_token_into_id(token_string, index):
    """Convert a token string to an integer ID."""
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    if last_token_start == -1:
        return None
    last_token = token_string[last_token_start:]
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    return None

# Sync function to be run in thread pool
def _orpheus_convert_to_audio_sync(multiframe, count):
    """Convert token frames to audio using SNAC (Synchronous version)."""
    if snac_model is None: return None
    if len(multiframe) < 7: return None

    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    for j in range(num_frames):
        i = 7 * j
        codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])
        codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 1]], device=snac_device, dtype=torch.int32)])
        codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device=snac_device, dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 2]], device=snac_device, dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device=snac_device, dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device=snac_device, dtype=torch.int32)])
        codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device=snac_device, dtype=torch.int32)])

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        # print(f"Warning: Invalid codes detected. Min/Max: c0({codes[0].min()},{codes[0].max()}), c1({codes[1].min()},{codes[1].max()}), c2({codes[2].min()},{codes[2].max()})")
        return None # Or handle appropriately

    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)

    # Return numpy array [2048,] (float32)
    audio_slice_np = audio_hat[:, :, 2048:4096].squeeze().cpu().numpy().astype(np.float32)
    return audio_slice_np

# --- Async Generators ---
async def generate_tokens_from_api_async(prompt, voice=DEFAULT_VOICE, temperature=0.6, top_p=0.9, max_tokens=1200, repetition_penalty=1.1):
    """Asynchronously generate token stream from the Orpheus model."""
    if llm is None:
        print("Error: Orpheus LLM not loaded.")
        yield "<|error|>" # Indicate error
        return

    formatted_prompt = format_prompt(prompt, voice)
    try:
        # Run the blocking Llama call in a separate thread
        response_stream = await asyncio.to_thread(
            llm,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repetition_penalty,
            stream=True
        )
        for data in response_stream:
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'text' in choice:
                    token_text = choice['text']
                    if token_text:
                        yield token_text
                # Small sleep to prevent blocking and allow other tasks
                await asyncio.sleep(0.001)
    except Exception as e:
        print(f"Error during Orpheus token generation: {e}")
        yield "<|error|>" # Indicate error

async def tokens_decoder_async(token_gen):
    """Asynchronously convert token stream to audio chunks (numpy arrays)."""
    buffer = []
    count = 0
    async for token_text in token_gen:
        if token_text == "<|error|>": # Check for error signal
             print("Error signal received from token generator.")
             break
        token = turn_token_into_id(token_text, count)
        if token is not None and token >= 0: # Allow token 0? Check Orpheus docs if 0 is valid. Assume >0 for now. Let's adjust to >=0 just in case.
            buffer.append(token)
            count += 1
            # Process buffer when enough tokens are collected (multiples of 7)
            # Adjust logic for buffering and processing based on SNAC requirements
            if count % 7 == 0 and count >= 28: # Need enough context, e.g., 4 frames = 28 tokens
                buffer_to_proc = buffer[-28:] # Process the last 28 tokens
                # Run sync SNAC conversion in thread pool
                audio_slice_np = await asyncio.to_thread(
                    _orpheus_convert_to_audio_sync, buffer_to_proc, count
                )
                if audio_slice_np is not None and audio_slice_np.size > 0:
                    yield audio_slice_np # Yield numpy array [2048,]
                # Small sleep to yield control
                await asyncio.sleep(0.001)

async def generate_audio_from_text_async(text, voice=DEFAULT_VOICE):
    """Generate audio chunks (sample_rate, numpy_array) from text as an async generator."""
    print(f"Generating audio for: '{text[:50]}...'")
    start_time = asyncio.get_event_loop().time()
    token_gen = generate_tokens_from_api_async(text, voice)
    chunk_count = 0
    async for audio_chunk_np in tokens_decoder_async(token_gen):
        yield (SAMPLE_RATE, audio_chunk_np)
        chunk_count += 1
    end_time = asyncio.get_event_loop().time()
    print(f"Audio generation finished. Yielded {chunk_count} chunks in {end_time - start_time:.2f}s")