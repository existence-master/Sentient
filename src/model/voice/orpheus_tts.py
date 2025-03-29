# orpheus_tts.py
import snac
import torch
import numpy as np
import asyncio
import threading
import queue
from llama_cpp import Llama

# Constants
MODEL_PATH = "models/orpheus-3b-0.1-ft-q4_k_m.gguf"
N_GPU_LAYERS = 30  # Adjust based on your GPU memory
SAMPLE_RATE = 24000
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Load SNAC model
snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"SNAC model using device: {snac_device}")
snac_model = snac_model.to(snac_device)

# Load Orpheus model with GPU support
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS)
print(f"Orpheus model loaded with {N_GPU_LAYERS} layers offloaded to GPU")

def format_prompt(prompt, voice=DEFAULT_VOICE):
    """Format the prompt with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
    formatted_prompt = f"{voice}: {prompt}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    return f"{special_start}{formatted_prompt}{special_end}"

def generate_tokens_from_api(prompt, voice=DEFAULT_VOICE, temperature=0.6, top_p=0.9, max_tokens=1200, repetition_penalty=1.1):
    """Generate token stream from the Orpheus model."""
    formatted_prompt = format_prompt(prompt, voice)
    response = llm(
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
        stream=True
    )
    for data in response:
        if 'choices' in data and len(data['choices']) > 0:
            choice = data['choices'][0]
            if 'text' in choice:
                token_text = choice['text']
                if token_text:
                    yield token_text

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

def orpheus_convert_to_audio(multiframe, count):
    """Convert token frames to audio using SNAC."""
    if len(multiframe) < 7:
        return None
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
        return None
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    audio_slice = audio_hat[:, :, 2048:4096]  # Return torch tensor [1, 1, 2048]
    return audio_slice

async def tokens_decoder(token_gen):
    """Asynchronously convert token stream to audio chunks."""
    buffer = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_slice = orpheus_convert_to_audio(buffer_to_proc, count)
                if audio_slice is not None:
                    yield audio_slice

def tokens_decoder_sync(syn_token_gen):
    """Synchronous wrapper for the asynchronous token decoder."""
    audio_queue = queue.Queue()

    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio
    thread.join()

def generate_audio_from_text(text, voice=DEFAULT_VOICE):
    """Generate audio chunks from text as a synchronous generator."""
    token_gen = generate_tokens_from_api(text, voice)
    return tokens_decoder_sync(token_gen)