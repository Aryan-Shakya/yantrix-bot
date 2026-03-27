import requests
import os
import wave

LAPTOP_IP = "10.101.196.138"
TEST_TEXT = "Hello, I am your robot assistant. Testing Piper text to speech."

print("=" * 50)
print("PIPER TTS QUICK TEST")
print("=" * 50)

# Step 1: Download audio from Piper server
print(f"\n[1] Downloading Piper audio from {LAPTOP_IP}:8000 ...")
try:
    response = requests.post(
        f"http://{LAPTOP_IP}:8000/api/tts",
        json={"text": TEST_TEXT},
        timeout=60
    )
    with open("test_audio.wav", "wb") as f:
        f.write(response.content)
    size = os.path.getsize("test_audio.wav")
    print(f"    Downloaded: {size} bytes")
    if size < 5000:
        print("    WARNING: File is too small! Piper failed on the laptop.")
        print("    Check your laptop_stt_server.py terminal for [TTS ERROR] messages!")
        exit()
except Exception as e:
    print(f"    FAILED: {e}")
    exit()

# Step 2: Read WAV file details
print("\n[2] WAV File Details:")
with wave.open("test_audio.wav", "rb") as w:
    print(f"    Channels   : {w.getnchannels()}")
    print(f"    Bit depth  : {w.getsampwidth()*8}-bit")
    print(f"    Sample Rate: {w.getframerate()} Hz")
    print(f"    Duration   : {w.getnframes()/w.getframerate():.2f} seconds")

# Step 3: Play using aplay (ALSA)
print("\n[3] Playing with aplay...")
ret = os.system("aplay test_audio.wav")
print(f"    Exit code: {ret}")
print("    Did you hear anything? (Yes/No)")

# Cleanup
if os.path.exists("test_audio.wav"):
    os.remove("test_audio.wav")
print("\n[DONE]")
