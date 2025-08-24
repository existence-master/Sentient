import asyncio
import json
import logging
from pydantic import BaseModel
import uuid
import wave
import os
import time
from datetime import datetime # Added import
from typing import AsyncGenerator, Dict, Any, Tuple
import re
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from functools import partial
from fastrtc import AlgoOptions, ReplyOnPause, SileroVadOptions, Stream, get_cloudflare_turn_credentials_async, get_cloudflare_turn_credentials
from fastrtc.utils import audio_to_float32, get_current_context

from main.auth.utils import AuthHelper
from main.dependencies import mongo_manager, auth_helper
from main.chat.utils import process_voice_command
from main.config import ENVIRONMENT, HF_TOKEN
from main.plans import PLAN_LIMITS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["Voice"])

rtc_token_cache: Dict[str, Dict] = {}
TOKEN_EXPIRATION_SECONDS = 600

# Define the Pydantic model for voice usage request
class VoiceUsageRequest(BaseModel):
    duration_seconds: int

@router.post("/initiate", summary="Initiate a voice chat session")
async def initiate_voice_session(
    user_id_and_plan: Tuple[str, str] = Depends(auth_helper.get_current_user_id_and_plan)
):
    """
    Generates a short-lived, single-use token for authenticating a WebRTC/WebSocket voice stream
    and provides the necessary ICE server configuration.
    """
    user_id, plan = user_id_and_plan
    logger.info(f"Initiating voice session for user: {user_id} on plan: {plan}")

    # --- Check Usage Limit ---
    usage = await mongo_manager.get_or_create_daily_usage(user_id)
    limit_seconds = PLAN_LIMITS[plan].get("voice_chat_daily_seconds", 0)
    used_seconds = usage.get("voice_chat_seconds", 0)

    if used_seconds >= limit_seconds:
        logger.warning(f"User {user_id} has exceeded their daily voice chat limit.")
        raise HTTPException(
            status_code=429,
            detail=f"You have used all of your daily voice chat time ({int(limit_seconds/60)} minutes). Please upgrade or try again tomorrow."
        )
    # --- End Limit Check ---

    # Clean up expired tokens to prevent the cache from growing indefinitely
    now = time.time()
    expired_tokens = [token for token, data in rtc_token_cache.items() if data["expires_at"] < now]
    for token in expired_tokens:
        del rtc_token_cache[token]

    # Generate a new token
    rtc_token = str(uuid.uuid4())
    expires_at = now + TOKEN_EXPIRATION_SECONDS
    rtc_token_cache[rtc_token] = {"user_id": user_id, "expires_at": expires_at}
    logger.info(f"Generated RTC token {rtc_token} for user {user_id}, expires at {datetime.fromtimestamp(expires_at).isoformat()}")
    
    if ENVIRONMENT in ["dev-local", "selfhost"]:
        logger.info(f"Initiated voice session for user {user_id} in dev-local mode with token {rtc_token}")
        return {"rtc_token": rtc_token, "ice_servers": []}  # No TURN server in dev-local mode
    
    else:
        logger.info(f"Initiated voice session for user {user_id} with token {rtc_token} using TURN server")
        # Get TURN credentials to send to the client
        ice_servers_config = await get_credentials()

        logger.info(f"Initiated voice session for user {user_id} with token {rtc_token}")
        return {"rtc_token": rtc_token, "ice_servers": ice_servers_config}


class MyVoiceChatHandler(ReplyOnPause):
    """
    A custom FastRTC handler for managing a real-time voice chat session.
    It orchestrates STT, LLM, and TTS in a fully streaming pipeline.
    """
    def __init__(self):
        # Initialize the parent ReplyOnPause class with VAD settings
        super().__init__(
            fn=self.process_audio_chunk,
            model_options=SileroVadOptions(
                threshold=0.6,  # Higher threshold for more aggressive VAD
                min_speech_duration_ms=250,
                min_silence_duration_ms=3000,   # wait 3s of silence
                speech_pad_ms=400,              # give extra buffer before and after speech
                max_speech_duration_s=15,
            ),
            algo_options=AlgoOptions(
                audio_chunk_duration=0.5,
                started_talking_threshold=0.2,
                speech_threshold=0.05,          # consider only more solid chunks as pause
            ),
            can_interrupt=False, # Set to False to prevent user interruption while bot is speaking,
            output_sample_rate=16000
        )
        self.handler_id = str(uuid.uuid4())[:8]
        logger.info(f"[Handler:{self.handler_id}] MyVoiceChatHandler instance created.")

    def copy(self):
        """Creates a new instance of the handler for each new connection."""
        logger.info(f"[Handler:{self.handler_id}] Copying handler for new connection.")
        new_handler = MyVoiceChatHandler()
        return new_handler

    async def process_audio_chunk(self, audio: tuple[int, np.ndarray]):
        """
        Main callback for FastRTC. Handles STT, LLM, and TTS streaming.
        This function is a generator, yielding audio chunks back to the client.
        """
        from main.app import stt_model_instance, tts_model_instance

        context = get_current_context()
        webrtc_id = context.webrtc_id
        logger.info(f"[Handler:{self.handler_id}] Processing audio chunk for webrtc_id: {webrtc_id}")

        # Authenticate the stream using the webrtc_id as the RTC token
        rtc_token = webrtc_id
        token_info = rtc_token_cache.get(rtc_token, None)

        if not token_info or time.time() > token_info["expires_at"]:
            logger.error(f"[Handler:{self.handler_id}] Invalid or expired RTC token received: {rtc_token}. Terminating stream.")
            await self.send_message(json.dumps({"type": "error", "message": "Authentication failed. Please refresh."}))
            return

        user_id = token_info["user_id"]
        logger.info(f"[Handler:{self.handler_id}] WebRTC stream authenticated for user {user_id} via token {rtc_token}")

        try:
            # 1. Speech-to-Text (STT)
            if not stt_model_instance:
                raise Exception("STT model is not initialized.")
            
            await self.send_message(json.dumps({"type": "status", "message": "transcribing"}))
            sample_rate, audio_array = audio

            # --- DEBUG: Save audio to file before STT ---
            try:
                debug_dir = "audio_debug_logs"
                # Create the directory relative to the current file's location
                server_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                full_debug_path = os.path.join(server_root, debug_dir)
                os.makedirs(full_debug_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(full_debug_path, f"stt_input_{timestamp}.wav")

                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(audio_array.dtype.itemsize) # Should be 2 for int16
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_array.tobytes())
                logger.info(f"[Handler:{self.handler_id}] DEBUG: Saved pre-STT audio to {filename}")
            except Exception as e:
                logger.error(f"[Handler:{self.handler_id}] DEBUG: Failed to save debug audio: {e}")
            # --- END DEBUG ---

            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).astype(np.int16)

            logger.debug(f"[Handler:{self.handler_id}] Transcribing audio for user {user_id}...")
            transcription, detected_language = await stt_model_instance.transcribe(audio_array.tobytes(), sample_rate=sample_rate)
            
            if not transcription or not transcription.strip():
                logger.info(f"[Handler:{self.handler_id}] STT returned empty string, skipping.")
                await self.send_message(json.dumps({"type": "status", "message": "listening"}))
                return

            logger.info(f"[Handler:{self.handler_id}] STT result for user {user_id}: '{transcription}' (Language: {detected_language})")
            await self.send_message(json.dumps({"type": "stt_result", "text": transcription, "language": detected_language}))

            async def send_status_update(status_update: Dict[str, Any]):
                """Sends a status update message to the client."""
                logger.info(f"[Handler:{self.handler_id}] Sending status update to client: {status_update}")
                await self.send_message(json.dumps(status_update))

            # 2. Process command, which may return a background task for Stage 2
            response_data = await process_voice_command(
                user_id=user_id,
                transcribed_text=transcription,
                detected_language=detected_language,
                send_status_update=send_status_update,
                db_manager=mongo_manager
            )

            interim_response = response_data.get("interim_response")
            final_response_task = response_data.get("final_response_task")
            final_response = response_data.get("final_response")
            assistant_message_id = response_data.get("assistant_message_id")

            # 3. Play interim TTS if available (runs in parallel with Stage 2)
            if interim_response:
                logger.info(f"[Handler:{self.handler_id}] TTS for interim response: '{interim_response}'")
                interim_audio_stream = tts_model_instance.stream_tts(interim_response, language=detected_language)
                async for audio_chunk in interim_audio_stream:
                    yield self._format_audio_chunk(audio_chunk)

            # 4. Wait for Stage 2 to finish (if it was started) and get final response
            if final_response_task:
                final_response_text, final_turn_steps = await final_response_task
                await mongo_manager.messages_collection.update_one(
                    {"message_id": assistant_message_id, "user_id": user_id},
                    {"$set": {"content": final_response_text, "turn_steps": final_turn_steps}}
                )
                await self.send_message(json.dumps({"type": "llm_result", "text": final_response_text, "messageId": assistant_message_id}))
                final_response = final_response_text
            elif final_response:
                await self.send_message(json.dumps({"type": "llm_result", "text": final_response, "messageId": assistant_message_id}))

            # 5. Play final TTS (handles all cases: simple, complex, conversational)
            if not final_response:
                logger.warning(f"[Handler:{self.handler_id}] LLM returned an empty final response for user {user_id}.")
                await self.send_message(json.dumps({"type": "status", "message": "listening"}))
                return

            await self.send_message(json.dumps({"type": "status", "message": "speaking"}))
            sentences = re.split(r'(?<=[.?!])\s+', final_response)
            sentences = [s.strip() for s in sentences if s.strip()]
            for i, sentence in enumerate(sentences):
                if not sentence: continue
                logger.info(f"[Handler:{self.handler_id}] Generating TTS for sentence {i+1}/{len(sentences)}: '{sentence}'")
                audio_stream = tts_model_instance.stream_tts(sentence, language=detected_language)
                chunk_count = 0
                async for audio_chunk in audio_stream:
                    yield self._format_audio_chunk(audio_chunk)
                    chunk_count += 1
                logger.info(f"[Handler:{self.handler_id}] Streamed {chunk_count} TTS chunks for sentence.")
        except Exception as e:
            logger.error(f"[Handler:{self.handler_id}] Error in voice_chat for user {user_id}: {e}", exc_info=True)
            await self.send_message(json.dumps({"type": "error", "message": str(e)}))
        finally:
            # Use a try-except block for the final message to prevent crashing on disconnect
            try:
                logger.info(f"[Handler:{self.handler_id}] Sending final 'listening' status for user {user_id}.")
                await self.send_message(json.dumps({"type": "status", "message": "listening"}))
            except Exception as final_e:
                logger.warning(f"[Handler:{self.handler_id}] Could not send final 'listening' status for user {user_id}, connection likely closed: {final_e}")
                
    def _format_audio_chunk(self, audio_chunk: Any) -> Tuple[int, np.ndarray]:
        """Helper to format audio chunks from different TTS providers into the expected tuple."""
        if isinstance(audio_chunk, tuple) and isinstance(audio_chunk[1], np.ndarray):
            # This is from Orpheus or SmallestAI TTS: (sample_rate, np.ndarray)
            sample_rate, audio_array = audio_chunk
            audio_float32 = audio_to_float32(audio_array)
            return (sample_rate, audio_float32)
        elif isinstance(audio_chunk, bytes):
            # This is from ElevenLabs TTS (PCM bytes)
            # Assuming 16kHz, 16-bit PCM from ElevenLabs
            sample_rate = 16000 
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_float32 = audio_to_float32(audio_array)
            return (sample_rate, audio_float32)
        raise TypeError(f"Unsupported audio chunk type: {type(audio_chunk)}")

stream = None

async def get_credentials():
    return await get_cloudflare_turn_credentials_async(hf_token=HF_TOKEN)

def get_server_credentials():
    creds = get_cloudflare_turn_credentials(hf_token=HF_TOKEN, ttl=360_000)
    logger.info("Using Cloudflare TURN server creds on server side: %s", creds)
    return creds

if ENVIRONMENT in ["dev-local", "selfhost"]:
    logger.info("Running in dev-local or selfhost mode, using no TURN server.")
    stream = Stream(
        handler=MyVoiceChatHandler(),
        modality="audio",
        mode="send-receive",
    )
else:
    logger.info("Using Cloudflare TURN server for WebRTC connections.")
    stream = Stream(
        handler=MyVoiceChatHandler(),
        rtc_configuration=get_credentials,
        server_rtc_configuration=get_server_credentials(),
        modality="audio",
        mode="send-receive",
    )


@router.post("/end", summary="End voice chat session")
async def end_voice_session(rtc_token: str):
    if rtc_token in rtc_token_cache:
        del rtc_token_cache[rtc_token]
        logger.info(f"RTC token {rtc_token} terminated and removed from cache.")
        return {"status": "terminated"}
    logger.warning(f"Attempted to end non-existent or already terminated RTC token: {rtc_token}")
    return {"status": "not_found"}

@router.post("/update-usage", summary="Update daily voice chat usage")
async def update_voice_usage(
    request: VoiceUsageRequest,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    logger.info(f"Updating voice usage for user {user_id} by {request.duration_seconds} seconds.")
    if request.duration_seconds < 0:
        raise HTTPException(status_code=400, detail="Invalid duration.")
    
    await mongo_manager.increment_daily_usage(user_id, "voice_chat_seconds", request.duration_seconds)
    logger.info(f"Usage updated successfully for user {user_id}.")
    return {"message": "Usage updated successfully."}