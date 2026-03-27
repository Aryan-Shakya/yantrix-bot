import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import requests
import wave

from faster_whisper import WhisperModel
from piper.voice import PiperVoice

app = FastAPI()

# ==========================================
# 1. LOAD WHISPER (HEARING)
# Using small.en — 3x faster than medium.en with great accuracy!
# ==========================================
print("==================================================")
print("1. Loading Faster-Whisper (small.en) onto CPU...")
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
print("   -> Whisper small.en Loaded!")

# ==========================================
# 2. LOAD PIPER (SPEAKING)
# ==========================================
print("\n2. Checking for Piper Robotic Voice files...")
VOICE_MODEL = "en_US-lessac-medium.onnx"
VOICE_JSON   = "en_US-lessac-medium.onnx.json"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"   -> Downloading: {dest} ...")
        r = requests.get(url, allow_redirects=True)
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"   -> Download complete!")
    else:
        print(f"   -> Found locally: {dest}")

download_file(
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    VOICE_MODEL
)
download_file(
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    VOICE_JSON
)

print("   -> Loading Piper voice into memory...")
piper_voice = PiperVoice.load(VOICE_MODEL)
print(f"   -> Piper loaded! Sample rate: {piper_voice.config.sample_rate} Hz")

# ==========================================
# 3. PRE-GENERATE PIPER FILLER AUDIO
# Pre-generate "Hmm, let me think..." once at startup for instant playback!
# ==========================================
FILLER_PATH = "filler_audio.wav"
print("\n3. Pre-generating filler audio...")
with wave.open(FILLER_PATH, "wb") as wf:
    piper_voice.synthesize_wav("Hmm, let me think...", wf)
print(f"   -> Filler audio ready! ({os.path.getsize(FILLER_PATH)} bytes)")

print("\n==================================================")
print("DUAL-BRAIN SERVER IS ONLINE!")
print("Waiting for Raspberry Pi on Port 8000...")
print("==================================================\n")


# ==========================================
# ENDPOINT 1: HEARING  /api/stt
# ==========================================
@app.post("/api/stt")
async def process_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as f:
        f.write(await file.read())

    print(f"[STT] Received audio. Transcribing...")
    segments, _ = whisper_model.transcribe(
        file_location,
        beam_size=3,                         # Reduced from 5 for speed
        condition_on_previous_text=False,
        language="en",
        vad_filter=True,                     # Skip silent parts faster!
        vad_parameters=dict(min_silence_duration_ms=300)
    )
    text = " ".join([s.text for s in segments]).strip()

    if os.path.exists(file_location):
        os.remove(file_location)

    print(f"[STT] '{text}'")
    return {"text": text}


# ==========================================
# ENDPOINT 2: SPEAKING  /api/tts
# ==========================================
class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def generate_speech(req: TTSRequest):
    print(f"[TTS] Generating: '{req.text[:80]}'")
    output_path = "output_speech.wav"

    try:
        with wave.open(output_path, "wb") as wav_file:
            piper_voice.synthesize_wav(req.text, wav_file)

        size = os.path.getsize(output_path)
        print(f"[TTS] Generated {size} bytes. Sending...")
        return FileResponse(output_path, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ==========================================
# ENDPOINT 3: FILLER AUDIO  /api/filler
# Pi downloads this once at startup for instant playback while thinking!
# ==========================================
@app.get("/api/filler")
async def get_filler_audio():
    return FileResponse(FILLER_PATH, media_type="audio/wav", filename="filler.wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
