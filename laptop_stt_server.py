import os
import wave
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# ── RAG engine
from rag_engine import build_knowledge_base, retrieve_context

app = FastAPI()

# ══════════════════════════════════════════════════════════════════
# 1. LOAD WHISPER (STT)
# ══════════════════════════════════════════════════════════════════
print("==================================================")
print("1. Loading Faster-Whisper (small.en) on CPU...")
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
print("   -> Whisper small.en loaded!")

# ══════════════════════════════════════════════════════════════════
# 2. LOAD PIPER (TTS)
# ══════════════════════════════════════════════════════════════════
import requests

VOICE_MODEL = "en_US-lessac-medium.onnx"
VOICE_JSON  = "en_US-lessac-medium.onnx.json"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"   -> Downloading: {dest} ...")
        r = requests.get(url, allow_redirects=True)
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"   -> Download complete!")
    else:
        print(f"   -> Found locally: {dest}")

print("\n2. Checking Piper voice files...")
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

# ══════════════════════════════════════════════════════════════════
# 3. PRE-GENERATE FILLER AUDIO
# ══════════════════════════════════════════════════════════════════
FILLER_PATH = "filler_audio.wav"
print("\n3. Pre-generating filler audio...")
with wave.open(FILLER_PATH, "wb") as wf:
    piper_voice.synthesize_wav("Hmm, let me think...", wf)
print(f"   -> Filler audio ready! ({os.path.getsize(FILLER_PATH)} bytes)")

# ══════════════════════════════════════════════════════════════════
# 4. BUILD RAG KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════
college_collection = build_knowledge_base()

print("\n==================================================")
print("YANTRIX BOT SERVER IS ONLINE — Port 8000")
print("RAG-powered | Whisper STT | Piper TTS | Ollama LLM")
print("==================================================\n")


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 1: STT + RAG  →  /api/stt
# Receives audio from Pi → transcribes → retrieves RAG context
# Returns both transcript and context to Pi
# ══════════════════════════════════════════════════════════════════
@app.post("/api/stt")
async def process_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location, "wb+") as f:
        f.write(await file.read())

    print(f"[STT] Received audio — transcribing...")
    segments, _ = whisper_model.transcribe(
        file_location,
        beam_size=3,
        condition_on_previous_text=False,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300)
    )
    transcript = " ".join([s.text for s in segments]).strip()

    if os.path.exists(file_location):
        os.remove(file_location)

    print(f"[STT] Transcript: '{transcript}'")

    # ── RAG retrieval
    context = ""
    if transcript:
        context = retrieve_context(college_collection, transcript)
        print(f"[RAG] Retrieved context ({len(context)} chars) for query: '{transcript[:60]}'")

    return JSONResponse(content={"text": transcript, "context": context})


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 2: TTS  →  /api/tts
# Receives text from Pi → synthesises with Piper → returns WAV
# ══════════════════════════════════════════════════════════════════
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
        print(f"[TTS] Generated {size} bytes — sending...")
        return FileResponse(output_path, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ══════════════════════════════════════════════════════════════════
# ENDPOINT 3: FILLER AUDIO  →  /api/filler
# Pi downloads this once at startup for instant playback
# ══════════════════════════════════════════════════════════════════
@app.get("/api/filler")
async def get_filler_audio():
    return FileResponse(FILLER_PATH, media_type="audio/wav", filename="filler.wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
