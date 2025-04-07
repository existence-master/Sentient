from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue


model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {snac_device}")
model = model.to(snac_device)


def orpheus_convert_to_audio(multiframe, count):
  frames = []
  if len(multiframe) < 7:
      print(f"Buffer too short: {len(multiframe)} tokens")
      return
  
  codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

  num_frames = len(multiframe) // 7
  frame = multiframe[:num_frames*7]

  for j in range(num_frames):
    i = 7*j
    if codes_0.shape[0] == 0:
      codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
    else:
      codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

    if codes_1.shape[0] == 0:
      
      codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    else:
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    
    if codes_2.shape[0] == 0:
      codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
    else:
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

  codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
  print(f"codes_0 range: {codes_0.min()} to {codes_0.max()}")
  print(f"codes_1 range: {codes_1.min()} to {codes_1.max()}")
  print(f"codes_2 range: {codes_2.min()} to {codes_2.max()}")  # Add these lines
  # check that all tokens are between 0 and 4096 otherwise return *
  if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
    return

  with torch.inference_mode():
    audio_hat = model.decode(codes)
  
  print(f"Audio shape after decoding: {audio_hat.shape}")  
  audio_slice = audio_hat[:, :, 2048:4096]
  detached_audio = audio_slice.detach().cpu()
  audio_np = detached_audio.numpy()
  audio_int16 = (audio_np * 32767).astype(np.int16)
  audio_bytes = audio_int16.tobytes()
  print(f"Generated audio bytes: {len(audio_bytes)}")
  return audio_bytes
  
    
async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    async for token_sim in token_gen:       
        token = turn_token_into_id(token_sim, count)
        if token is None:
            pass
        else:
            if token > 0:
                buffer.append(token)
                count += 1

                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples


# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):

    audio_queue = queue.Queue()

    # Convert the synchronous token generator into an async generator.
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        # tokens_decoder.tokens_decoder is assumed to be an async generator that processes tokens.
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel

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

import os
import sys
import json
import time
import wave
import numpy as np
import sounddevice as sd
import argparse
import threading
import queue
import asyncio
from llama_cpp import Llama

# Set the path to your GGUF model file (update this to the correct path)
MODEL_PATH = "./models/orpheus-3b-0.1-ft-q4_k_m.gguf"  # Replace with your GGUF file path

# Number of layers to offload to GPU (adjust based on your GPU memory, e.g., 30 for 8GB VRAM)
N_GPU_LAYERS = 20

# Load the model with GPU support
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=N_GPU_LAYERS)

# Model parameters
MAX_TOKENS = 1200
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1
SAMPLE_RATE = 24000  # SNAC model uses 24kHz

# Available voices based on the Orpheus-TTS repository
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Special token IDs for Orpheus model (used in token processing)
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"

def format_prompt(prompt, voice=DEFAULT_VOICE):
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config
    
    return f"{special_start}{formatted_prompt}{special_end}"

def generate_tokens_from_api(prompt, voice=DEFAULT_VOICE, temperature=TEMPERATURE, 
                            top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
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
                    print(f"Generated token: '{token_text}'")  # Add this line
                    yield token_text
            else:
                print("Warning: 'text' key not found in choice")
        else:
            print("Warning: 'choices' key not found or empty in response data")



def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        print(f"No custom token found in: '{token_string}'")
        return None
    
    last_token = token_string[last_token_start:]
    print(f"Processing token: '{last_token}'")  # Add this line
    
    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            print(f"Extracted ID: {token_id} from '{last_token}' at index {index}")  # Add this line
            return token_id
        except ValueError:
            print(f"Failed to parse number from '{last_token}'")
            return None
    else:
        print(f"Token '{last_token}' does not match expected format")
        return None


def convert_to_audio(multiframe, count):
    """Convert token frames to audio."""
    # Import here to avoid circular imports
    return orpheus_convert_to_audio(multiframe, count)

async def tokens_decoder(token_gen):
    """Asynchronous token decoder that converts token stream to audio stream."""
    buffer = []
    count = 0
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """Synchronous wrapper for the asynchronous token decoder."""
    audio_queue = queue.Queue()
    audio_segments = []
    
    # If output_file is provided, prepare WAV file
    wav_file = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    # Convert the synchronous token generator into an async generator
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)  # Sentinel to indicate completion

    def run_async():
        asyncio.run(async_producer())

    # Start the async producer in a separate thread
    thread = threading.Thread(target=run_async)
    thread.start()

    # Process audio as it becomes available
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        print(f"Audio segment received, length: {len(audio)} bytes")  
        audio_segments.append(audio)
        
        # Write to WAV file if provided
        if wav_file:
            wav_file.writeframes(audio)
    
    # Close WAV file if opened
    if wav_file:
        wav_file.close()
    
    thread.join()
    
    # Calculate and print duration
    duration = sum([len(segment) // (2 * 1) for segment in audio_segments]) / SAMPLE_RATE
    print(f"Generated {len(audio_segments)} audio segments")
    print(f"Generated {duration:.2f} seconds of audio")
    
    return audio_segments

def stream_audio(audio_buffer):
    """Stream audio buffer to output device."""
    if audio_buffer is None or len(audio_buffer) == 0:
        return
    
    # Convert bytes to NumPy array (16-bit PCM)
    audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
    
    # Normalize to float in range [-1, 1] for playback
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    # Play the audio
    sd.play(audio_float, SAMPLE_RATE)
    sd.wait()

def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None, temperature=TEMPERATURE, 
                     top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=REPETITION_PENALTY):
    """Generate speech from text using Orpheus model locally."""
    return tokens_decoder_sync(
        generate_tokens_from_api(
            prompt=prompt, 
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        ),
        output_file=output_file
    )

def list_available_voices():
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")
    
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech using local GGUF model")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, 
                       help="Repetition penalty (>=1.1 required for stable generation)")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Use text from command line or prompt user
    prompt = args.text
    if not prompt:
        if len(sys.argv) > 1 and sys.argv[1] not in ("--voice", "--output", "--temperature", "--top_p", "--repetition_penalty"):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."
    
    # Default output file if none provided
    output_file = args.output
    if not output_file:
        os.makedirs("outputs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")
    
    # Generate speech
    start_time = time.time()
    audio_segments = generate_speech_from_api(
        prompt=prompt,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_file=output_file
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    main()