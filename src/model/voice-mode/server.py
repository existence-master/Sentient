from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import speech_recognition as sr
from gtts import gTTS
import httpx
import wave
import httpx
import json
import os
import uvicorn
import nest_asyncio
from pydub import AudioSegment
import re
import io
import numpy as np

nest_asyncio.apply()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceChatRequest(BaseModel):
    chat_id: str

async def process_audio(audio_buffer: io.BytesIO, chat_id: str):
    """Process audio buffer: transcribe, get chat response, generate audio."""
    temp_file = "temp_audio.webm"
    wav_file = "temp_audio.wav"
    try:
        # Setting Chat ID on Orchestrator
        set_chat_url = "http://localhost:5000/set-chat"
        set_chat_payload = {"id": chat_id}
        print("Setting chat ID on Orchestrator:" + chat_id)
        async with httpx.AsyncClient() as client:
            set_chat_response = await client.post(set_chat_url, json=set_chat_payload)
            if set_chat_response.status_code != 200:
                print(f"Failed to set chat ID: {set_chat_response.text}")
                raise Exception(f"Failed to set chat ID: {set_chat_response.status_code}")

        # Save buffer to temporary file
        audio_buffer.seek(0)
        with open(temp_file, "wb") as f:
            f.write(audio_buffer.getvalue())

        # Convert to WAV
        audio_segment = AudioSegment.from_file(temp_file)
        if audio_segment.duration_seconds < 0.5:
            raise HTTPException(status_code=400, detail="Audio too short for transcription")
        audio_segment.export(wav_file, format="wav")

        # Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
            try:
                query = recognizer.recognize_google(audio_data)
                print(f"Transcribed query: {query}")
            except sr.UnknownValueError:
                query = ""
                print("Could not understand audio")

        # Prepare payload for chat service
        payload = {
            "input": query,
            "pricing": "pro",
            "credits": 0,
            "chat_id": chat_id
        }
        chat_url = "http://localhost:5000/chat"

        # Fetch chat response
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(chat_url, json=payload)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            full_text = ""
            try:
                data = response.json()
                full_text = data.get("message", "")
            except json.JSONDecodeError:
                lines = response.text.splitlines()
                for line in lines:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            token = data.get("token", "")
                            full_text += token
                        except json.JSONDecodeError:
                            continue
            if not full_text:
                raise HTTPException(status_code=500, detail="No response text received from chat service")

        # Generate audio from response using gTTS
        sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
        audio_data_list = []
        for sentence in sentences:
            if sentence.strip():
                tts = gTTS(text=sentence, lang='en')
                with io.BytesIO() as mp3_io:
                    tts.write_to_fp(mp3_io)
                    mp3_io.seek(0)
                    audio_data_list.append(mp3_io.read())
        combined_mp3 = b''.join(audio_data_list)

        # Convert combined MP3 to WAV
        with io.BytesIO(combined_mp3) as mp3_io:
            audio_segment = AudioSegment.from_mp3(mp3_io)
            with io.BytesIO() as wav_io:
                audio_segment.export(wav_io, format="wav")
                audio_bytes = wav_io.getvalue()

        return query, full_text, audio_bytes

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(wav_file):
            os.remove(wav_file)



@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...), chat_id: str = Form(...)):
    print(f"Received chat_id: {chat_id}")
    print(f"Received audio file: {audio.filename}")
    
    audio_buffer = io.BytesIO(await audio.read())
    query, full_text, audio_bytes = await process_audio(audio_buffer, chat_id)
    
    headers = {
        "X-Transcription": query,
        "X-Response": full_text[:100] + ("..." if len(full_text) > 100 else "")
    }
    
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers=headers
    )

@app.websocket("/voice-chat-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = io.BytesIO()
    chat_id = None
    connection_open = True  # Flag to track connection state

    try:
        while connection_open:
            data = await websocket.receive()
            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    print(f"Received audio chunk of size: {len(data['bytes'])} bytes")
                    audio_buffer.write(data['bytes'])
                elif "text" in data:
                    if data["text"] == "END":
                        print("Received END message, processing audio...")
                        query, full_text, audio_bytes = await process_audio(audio_buffer, chat_id)
                        # Send structured messages as JSON
                        await websocket.send_text(json.dumps({
                            "messages": [
                                {"type": "userMessage", "message": query, "isUser": True},
                                {"type": "assistantMessage", "message": full_text, "isUser": False, "memoryUsed": False, "agentsUsed": False, "internetUsed": False}
                            ]
                        }))
                        await websocket.send_bytes(audio_bytes)
                        await websocket.close()  # Close after sending response
                        connection_open = False  # Update flag
                    else:
                        chat_id = data["text"]
                        print(f"Received chat_id: {chat_id}")
    except WebSocketDisconnect:
        print("WebSocket disconnected unexpectedly")
        connection_open = False
    except Exception as e:
        print(f"Error: {e}")
        if connection_open:  # Only close if still open
            await websocket.close()
            connection_open = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5008)