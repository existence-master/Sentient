from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import speech_recognition as sr
from TTS.api import TTS as CoquiTTS
import io
import wave
import httpx
import json
import os
import uvicorn
import nest_asyncio

nest_asyncio.apply()

app = FastAPI()
tts_model = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

class VoiceChatRequest(BaseModel):
    chat_id: str

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...), chat_id: str = Form(...)):
    temp_file = "temp_audio.wav"
    try:
        with open(temp_file, "wb") as f:
            f.write(await audio.read())
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_file) as source:
            audio_data = recognizer.record(source)
            try:
                query = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                query = ""
            except sr.RequestError as e:
                raise HTTPException(status_code=500, detail=f"Speech recognition error: {e}")
        
        payload = {
            "input": query,
            "pricing": "free",
            "credits": 0,
            "chat_id": chat_id
        }
        chat_url = "http://localhost:5000/chat"

        async def generate_audio():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", chat_url, json=payload) as response:
                    if response.status_code != 200:
                        yield f"Error: {response.status_code} - {response.text}".encode()
                        return
                    sentence = ""
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            token = data.get("token", "")
                            sentence += token
                            if token in [".", "!", "?"]:
                                audio_data = tts_model.tts(text=sentence)
                                with io.BytesIO() as wav_io:
                                    with wave.open(wav_io, 'wb') as wav_file:
                                        wav_file.setnchannels(1)
                                        wav_file.setsampwidth(2)
                                        wav_file.setframerate(22050)
                                        wav_file.writeframes(audio_data)
                                    yield wav_io.getvalue()
                                sentence = ""

        return StreamingResponse(generate_audio(), media_type="audio/wav")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5008)