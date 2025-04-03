import numpy as np
import asyncio
import httpx
import traceback
import uvicorn
import json # Import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect # Import WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Import fastrtc components
from fastrtc import Stream, ReplyOnPause, AlgoOptions, SileroVadOptions

# Import your STT and TTS models
from stt import FasterWhisperSTT
from orpheus_tts import OrpheusTTS, TTSOptions, VoiceId, AVAILABLE_VOICES

# ... (Configuration, Model Init, Ollama function remain the same) ...
# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_REQUEST_TIMEOUT = 60.0

# --- Select TTS Voice ---
SELECTED_TTS_VOICE: VoiceId = "tara"
if SELECTED_TTS_VOICE not in AVAILABLE_VOICES:
    print(f"Warning: Selected voice '{SELECTED_TTS_VOICE}' not valid. Using default 'tara'.")
    SELECTED_TTS_VOICE = "tara"
# --- End Voice Selection ---


# --- Orpheus Supported Emotion Tags ---
ORPHEUS_EMOTION_TAGS = [
    "<giggle>", "<laugh>", "<chuckle>", "<sigh>", "<cough>",
    "<sniffle>", "<groan>", "<yawn>", "<gasp>"
]
EMOTION_TAG_LIST_STR = ", ".join(ORPHEUS_EMOTION_TAGS)

# --- Model Initialization ---
print("Loading STT model...")
stt_model = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
print("STT model loaded.")

print("Loading TTS model...")
try:
    tts_model = OrpheusTTS(
        verbose=False,
        default_voice_id=SELECTED_TTS_VOICE
    )
    print("TTS model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load TTS model: {e}")
    exit(1)

# --- Ollama API Call Function (remains the same) ---
async def get_ollama_response_with_emotions(user_text: str) -> str | None:
    # ... (function code is unchanged) ...
    print(f"Sending to Ollama ({OLLAMA_MODEL}) with emotion instructions: '{user_text}'")
    system_prompt = (
        "You are a helpful and expressive assistant. The text responses you generate are going to be passed to a speech generation model."
        "This speech generation model supports the usage of emotion tags to generate expressive audio. You can use these special tags to insert non-verbal sounds and express emotions in your response."
        f"The available tags are: {EMOTION_TAG_LIST_STR}. "
        "ONLY FOLLOW THIS SYNTAX FOR THE TAGS: <tag>. DO NOT USE ANY OTHER SYNTAX LIKE *sigh* OR (sigh). Do not add visual cues like *big smile* or (smiling). The user can't see you smiling."
        "Insert these tags naturally within your sentences where appropriate to convey emotion. "
        "For example: 'Oh, <gasp> that's surprising!' or 'Well, <sigh> I understand.' or 'That's quite funny! <chuckle>'. "
        "Use them sparingly and only when it genuinely enhances the emotional tone of the response. "
        "Do not make up tags."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            bot_response = data.get("message", {}).get("content")
            if bot_response:
                print(f"Ollama response (with potential tags): '{bot_response}'")
                if any(tag in bot_response for tag in ORPHEUS_EMOTION_TAGS):
                    print("Ollama response included emotion tags.")
                return bot_response.strip()
            else:
                print("Warning: Ollama response format unexpected or empty content.")
                print(f"Full Ollama response data: {data}")
                return None
    except httpx.RequestError as exc:
        print(f"Error calling Ollama API: {exc}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Ollama API call: {e}")
        traceback.print_exc()
        return None


# --- Define the Async Handler (remains the same, receives np.ndarray) ---
async def handle_audio_conversation(audio: tuple[int, np.ndarray]):
    # ... (function code is unchanged, yields (sr, np.ndarray)) ...
    print("\n--- Received audio (via WebSocket processing by fastrtc) ---") # Updated log hint
    print("Transcribing...")
    user_text = stt_model.stt(audio)
    if not user_text or not user_text.strip():
        print("No valid text transcribed, skipping.")
        return

    print(f"User: {user_text}") # Log user text to console

    bot_response_text = await get_ollama_response_with_emotions(user_text)
    if not bot_response_text:
        print("No response received from Ollama, skipping TTS.")
        print("[LLM Error or No Response]") # Log error to console
        return

    print(f"Bot: {bot_response_text}") # Log bot text to console

    print(f"Synthesizing speech (default voice: {tts_model.instance_default_voice})...")
    try:
        tts_options: TTSOptions = {} # Use instance default voice

        chunk_count = 0
        async for (sample_rate, audio_chunk) in tts_model.stream_tts(bot_response_text, options=tts_options):
            if chunk_count == 0:
                 print(f"Streaming TTS audio chunk 1 (sr={sample_rate}, shape={audio_chunk.shape}, dtype={audio_chunk.dtype})...")
            # fastrtc handles sending this back over the WebSocket connection
            yield (sample_rate, audio_chunk)
            chunk_count += 1

        print(f"Finished synthesizing and streaming {chunk_count} chunks.")

    except Exception as e:
        print(f"Error during TTS synthesis or streaming: {e}")
        traceback.print_exc()
        print("[TTS Error]") # Log error to console

# --- Set up the Stream (Handler remains the same) ---
stream = Stream(
    ReplyOnPause(
        handle_audio_conversation,
        # Keep VAD settings as they worked in Gradio
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.25,
            speech_threshold=0.2
        ),
        model_options=SileroVadOptions(
            threshold=0.5,
            min_speech_duration_ms=200,
            min_silence_duration_ms=1000
        ),
        can_interrupt=False,
    ),
    mode="send-receive",
    modality="audio",
    additional_outputs=None,
    additional_outputs_handler=None
)

# --- Setup FastAPI App ---
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "app://.",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\nMounting FastRTC stream...")
# Mount under '/voice'. This should handle BOTH WebRTC and WebSocket endpoints.
# WebRTC Offer/Answer: /voice/webrtc/offer
# WebSocket: /voice/websocket/offer (based on docs) or potentially /voice/ws
stream.mount(app, path="/voice")
print("FastRTC stream mounted at /voice.")
print("Expected WebSocket endpoint: ws://<host>:<port>/voice/websocket/offer")

@app.get("/")
async def root():
    return {"message": "FastRTC server is running. Connect via WebSocket or WebRTC."}

# --- Run the server ---
if __name__ == "__main__":
    print(f"\nStarting Uvicorn server...")
    # ... (other print statements) ...
    print(f"TTS Default Voice: {SELECTED_TTS_VOICE}")
    print(f"Access API at http://0.0.0.0:8000")
    print(f"WebSocket endpoint expected at /voice/websocket/offer")

    uvicorn.run(app, host="0.0.0.0", port=8000)