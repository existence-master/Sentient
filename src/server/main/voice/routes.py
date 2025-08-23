import asyncio
import json
import logging
from pydantic import BaseModel
import uuid
import time
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

    # --- Check Usage Limit ---
    usage = await mongo_manager.get_or_create_daily_usage(user_id)
    limit_seconds = PLAN_LIMITS[plan].get("voice_chat_daily_seconds", 0)
    used_seconds = usage.get("voice_chat_seconds", 0)

    if used_seconds >= limit_seconds:
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
                threshold=0.9,  # Higher threshold for more aggressive VAD
                min_speech_duration_ms=250,
                min_silence_duration_ms=3000,   # wait 3s of silence
                speech_pad_ms=800,              # give extra buffer before and after speech
                max_speech_duration_s=15,
            ),
            algo_options=AlgoOptions(
                audio_chunk_duration=0.5,
                started_talking_threshold=0.2,
                speech_threshold=0.05,          # consider only more solid chunks as pause
            ),
            can_interrupt=False, # Set to False to prevent user interruption while bot is speaking
        )

    def copy(self):
        """Creates a new instance of the handler for each new connection."""
        return MyVoiceChatHandler()

    async def process_audio_chunk(self, audio: tuple[int, np.ndarray]):
        """
        Main callback for FastRTC. Handles STT, LLM, and TTS streaming.
        This function is a generator, yielding audio chunks back to the client.
        """
        from main.app import stt_model_instance, tts_model_instance

        context = get_current_context()
        webrtc_id = context.webrtc_id

        # Authenticate the stream using the webrtc_id as the RTC token
        rtc_token = webrtc_id
        token_info = rtc_token_cache.get(rtc_token, None)

        if not token_info or time.time() > token_info["expires_at"]:
            logger.error(f"Invalid or expired RTC token received: {rtc_token}. Terminating stream.")
            await self.send_message(json.dumps({"type": "error", "message": "Authentication failed. Please refresh."}))
            return

        user_id = token_info["user_id"]
        logger.info(f"WebRTC stream authenticated for user {user_id} via token {rtc_token}")

        try:
            # 1. Speech-to-Text (STT)
            if not stt_model_instance:
                raise Exception("STT model is not initialized.")
            
            await self.send_message(json.dumps({"type": "status", "message": "transcribing"}))
            sample_rate, audio_array = audio
            
            if audio_array.dtype != np.int16:
                audio_array = (audio_array * 32767).astype(np.int16)

            transcription, detected_language = await stt_model_instance.transcribe(audio_array.tobytes(), sample_rate=sample_rate)
            
            if not transcription or not transcription.strip():
                logger.info("STT returned empty string, skipping.")
                await self.send_message(json.dumps({"type": "status", "message": "listening"}))
                return

            logger.info(f"STT result for user {user_id}: '{transcription}' (Language: {detected_language})")
            await self.send_message(json.dumps({"type": "stt_result", "text": transcription, "language": detected_language}))

            # 2. FULL AGENTIC LLM PROCESSING
            # Define the callback function that process_voice_command will use to send status updates
            async def send_status_update(status_update: Dict[str, Any]):
                """Sends a status update message to the client."""
                await self.send_message(json.dumps(status_update))

            # Call the updated, fully-featured voice command processor
            full_response_buffer, assistant_message_id = await process_voice_command(
                user_id=user_id,
                transcribed_text=transcription,
                detected_language=detected_language,
                send_status_update=send_status_update,
                db_manager=mongo_manager
            )
            
            await self.send_message(json.dumps({"type": "llm_result", "text": full_response_buffer, "messageId": assistant_message_id}))

            # 3. Text-to-Speech (TTS) per sentence
            if not tts_model_instance:
                raise Exception("TTS model is not initialized.")
            if not full_response_buffer:
                logger.warning(f"LLM returned an empty response for user {user_id}.")
                await self.send_message(json.dumps({"type": "status", "message": "listening"}))
                return

            await self.send_message(json.dumps({"type": "status", "message": "speaking"}))
            
            sentences = re.split(r'(?<=[.?!])\s+', full_response_buffer)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for sentence in sentences:
                if not sentence: continue
                logger.info(f"Generating TTS for sentence: '{sentence}'")
                audio_stream = tts_model_instance.stream_tts(sentence, language=detected_language)
                
                async for audio_chunk in audio_stream:
                    if isinstance(audio_chunk, tuple) and isinstance(audio_chunk[1], np.ndarray):
                        # This is from Orpheus or SmallestAI TTS: (sample_rate, np.ndarray)
                        sample_rate, audio_array = audio_chunk
                        audio_float32 = audio_to_float32(audio_array)
                        yield (sample_rate, audio_float32)
                    elif isinstance(audio_chunk, bytes):
                        # This is from ElevenLabs TTS (PCM bytes)
                        # We need to convert it to the format fastrtc expects: (sample_rate, np.ndarray)
                        # Assuming 16kHz, 16-bit PCM from ElevenLabs
                        sample_rate = 16000 
                        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                        audio_float32 = audio_to_float32(audio_array)
                        yield (sample_rate, audio_float32)

        except Exception as e:
            logger.error(f"Error in voice_chat for user {user_id}: {e}", exc_info=True)
            await self.send_message(json.dumps({"type": "error", "message": str(e)}))
        finally:
            # Use a try-except block for the final message to prevent crashing on disconnect
            try:
                await self.send_message(json.dumps({"type": "status", "message": "listening"}))
            except Exception as final_e:
                logger.warning(f"Could not send final 'listening' status for user {user_id}, connection likely closed: {final_e}")
                
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
        return {"status": "terminated"}
    return {"status": "not_found"}

@router.post("/update-usage", summary="Update daily voice chat usage")
async def update_voice_usage(
    request: VoiceUsageRequest,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    if request.duration_seconds < 0:
        raise HTTPException(status_code=400, detail="Invalid duration.")
    
    await mongo_manager.increment_daily_usage(user_id, "voice_chat_seconds", request.duration_seconds)
    return {"message": "Usage updated successfully."}