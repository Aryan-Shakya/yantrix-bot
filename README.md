# 🤖 AI Voice Bot — Offline RTX-Powered Robot Assistant

A fully offline AI voice assistant built on a **Raspberry Pi 4B** + **Windows Laptop (RTX 4050)** architecture.

**Speak → Hear → Think → Speak Back** — completely privately, no cloud required.

---

## Architecture

```
Raspberry Pi 4B (Client)              Windows Laptop (Server)
┌──────────────────────┐              ┌─────────────────────────┐
│  USB Mic → Record    │──WAV file──▶ │  Whisper STT (offline)  │
│                      │◀──text──────│                         │
│  Text → Ask Ollama   │──text──────▶│  Ollama llama3.2 (LLM)  │
│                      │◀──answer────│                         │
│  Answer → Play Audio │──text──────▶│  Piper TTS (offline)    │
│  Bluetooth Speaker   │◀──WAV file──│                         │
└──────────────────────┘              └─────────────────────────┘
```

## Features
- 🎤 **Offline STT** — Faster-Whisper (`small.en` on CPU)
- 🧠 **Offline LLM** — Ollama `llama3.2` (3B model, GPU accelerated)
- 🔊 **Offline TTS** — Piper TTS with `en_US-lessac-medium` robotic voice
- 💬 **Conversation Memory** — Remembers last 5 exchanges
- ⚡ **Sentence Streaming** — Speaks first sentence before full response is ready

---

## Hardware Requirements

| Device | Role |
|--------|------|
| Raspberry Pi 4B | Client (ears + mouth) |
| Windows Laptop with RTX GPU | Server (brain) |
| USB Webcam/Microphone | Audio input on Pi |
| Bluetooth Speaker | Audio output on Pi |

---

## Quick Setup

### Laptop (Server)

**1. Install dependencies:**
```bash
pip install fastapi uvicorn faster-whisper python-multipart piper-tts
```

**2. Install Ollama and pull the model:**
```bash
# Download from https://ollama.ai
ollama pull llama3.2
```

**3. Start servers:**
```cmd
# Terminal 1 — LLM Brain
set OLLAMA_HOST=0.0.0.0
ollama serve

# Terminal 2 — STT + TTS Server
python laptop_stt_server.py
```

> **Note:** The Piper voice model (`en_US-lessac-medium.onnx`, ~63MB) is automatically downloaded on first launch.

---

### Raspberry Pi (Client)

**1. Install dependencies:**
```bash
sudo apt install -y python3-pip portaudio19-dev alsa-utils
pip3 install speechrecognition pyaudio requests
```

**2. Edit `pi_client.py` — set your laptop's IP:**
```python
LAPTOP_IP = "YOUR_LAPTOP_IP"  # Run 'ipconfig' on your laptop
```

**3. Pair your Bluetooth speaker, select it from the audio tray, then run:**
```bash
python3 pi_client.py
```

---

## Files

| File | Description |
|------|-------------|
| `laptop_stt_server.py` | FastAPI server — Whisper STT + Piper TTS endpoints |
| `pi_client.py` | Raspberry Pi client — Mic → Whisper → Ollama → Piper → Speaker |
| `laptop_bot.py` | Standalone local test script (laptop only) |
| `test_piper.py` | Pi-side test script to verify Piper audio |
| `test_piper_laptop.py` | Laptop-side Piper diagnostic script |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Cannot reach laptop | Update `LAPTOP_IP` in `pi_client.py` (IP changes with network) |
| Ollama timeout | Run `set OLLAMA_HOST=0.0.0.0` before `ollama serve` |
| No audio on Pi | Click taskbar audio icon → select Bluetooth speaker |
| Port blocked | Allow ports 8000 and 11434 in Windows Firewall |
| Empty STT transcription | Speak louder / closer to the microphone |
