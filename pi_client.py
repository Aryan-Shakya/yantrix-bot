import speech_recognition as sr
import requests
import re
import os
import subprocess
import threading

# ==========================================
# IMPORTANT: Put your laptop's actual Wi-Fi IP address here!
LAPTOP_IP = "10.101.196.138"
# ==========================================

OLLAMA_URL   = f"http://{LAPTOP_IP}:11434/api/chat"
WHISPER_URL  = f"http://{LAPTOP_IP}:8000/api/stt"
TTS_URL      = f"http://{LAPTOP_IP}:8000/api/tts"
FILLER_URL   = f"http://{LAPTOP_IP}:8000/api/filler"

FILLER_FILE  = "/home/yantrix/filler_audio.wav"  # Downloaded once at startup

# Conversation memory
memory = [
    {"role": "system", "content": "You are a helpful, concise, and conversational AI robot assistant. Keep responses under 3 sentences."}
]

# ==========================================
# 1. DOWNLOAD FILLER AUDIO AT STARTUP
# ==========================================
def download_filler():
    try:
        r = requests.get(FILLER_URL, timeout=10)
        with open(FILLER_FILE, "wb") as f:
            f.write(r.content)
        print(f"   -> Filler audio ready ({os.path.getsize(FILLER_FILE)} bytes)")
    except Exception as e:
        print(f"   -> Could not download filler: {e}")

# ==========================================
# 2. TTS - Speak via Piper on Laptop
# ==========================================
def speak(text):
    print(f"Bot: {text}")
    try:
        response = requests.post(TTS_URL, json={"text": text}, timeout=60)
        if response.status_code == 200:
            filename = "/tmp/response_audio.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            os.system(f"aplay {filename} 2>/dev/null")
            os.remove(filename)
        else:
            print(f"   [Speaker ERROR] Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"   [Speaker ERROR] Cannot reach laptop!")
    except requests.exceptions.Timeout:
        print("   [Speaker ERROR] TTS timed out.")
    except Exception as e:
        print(f"   [Speaker ERROR] {e}")

def play_filler():
    """Play the pre-downloaded filler sound in a background thread."""
    if os.path.exists(FILLER_FILE):
        subprocess.Popen(["aplay", FILLER_FILE],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

# ==========================================
# 3. BRAIN - Ask Ollama with streaming
# Sends text → gets streamed response → speaks sentence-by-sentence
# ==========================================
def ask_and_speak(prompt):
    global memory
    memory.append({"role": "user", "content": prompt})

    data = {
        "model": "llama3.2",
        "messages": memory,
        "stream": True,            # Enable streaming for faster first response!
        "keep_alive": -1
    }

    try:
        print("   [Brain] Streaming response from Ollama...")
        response = requests.post(OLLAMA_URL, json=data, stream=True, timeout=30)

        full_reply = ""
        sentence_buffer = ""
        first_sentence_spoken = False

        for line in response.iter_lines():
            if not line:
                continue
            import json
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            sentence_buffer += token
            full_reply += token

            # Speak as soon as a complete sentence is ready!
            if any(sentence_buffer.rstrip().endswith(p) for p in [".", "!", "?", "...", ":"]):
                sentence = sentence_buffer.strip()
                sentence_buffer = ""

                if sentence:
                    # Clean any markdown
                    sentence = re.sub(r'[*#_`]', '', sentence)
                    sentence = re.sub(r'```.*?```', '[code]', sentence, flags=re.DOTALL)

                    if not first_sentence_spoken:
                        first_sentence_spoken = True
                    speak(sentence)

            if chunk.get("done"):
                break

        # Speak any remaining text
        if sentence_buffer.strip():
            leftover = re.sub(r'[*#_`]', '', sentence_buffer.strip())
            if leftover:
                speak(leftover)

        # Save to memory
        memory.append({"role": "assistant", "content": full_reply})
        if len(memory) > 11:
            memory = [memory[0]] + memory[-10:]

    except requests.exceptions.ConnectionError:
        speak("I cannot connect to the brain. Is Ollama running?")
    except Exception as e:
        speak("Sorry, there was an error.")
        print(f"   [Brain ERROR] {e}")

# ==========================================
# 4. EARS - Listen via Whisper on Laptop
# ==========================================
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.8         # Wait 0.8s of silence before stopping
recognizer.non_speaking_duration = 0.5  # Consistent with old working version

def listen():
    with sr.Microphone() as source:
        print("\nAdjusting to background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)  # Full 1s for accuracy
        print("Listening... (Speak loudly into your Lenovo webcam mic)")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

            with open("/tmp/temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            with open("/tmp/temp_audio.wav", "rb") as f:
                r = requests.post(WHISPER_URL, files={"file": f}, timeout=20)

            text = r.json().get("text", "").strip()
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("[STT] Whisper returned empty. Speak louder!")
                return None

        except sr.WaitTimeoutError:
            print("[STT] No speech detected.")
            return None
        except requests.exceptions.ConnectionError:
            print(f"[STT ERROR] Cannot reach laptop at {LAPTOP_IP}:8000!")
            return None
        except requests.exceptions.Timeout:
            print("[STT ERROR] Whisper took too long.")
            return None
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return None

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("Downloading filler audio from laptop...")
    download_filler()

    speak("Raspberry Pi body is online. Connecting to the RTX Brain.")

    while True:
        user_input = listen()
        if not user_input:
            continue

        if any(w in user_input.lower() for w in ["stop", "exit", "shutdown", "bye"]):
            speak("Shutting down. Goodbye!")
            break

        # Play filler instantly while thinking (non-blocking)
        play_filler()

        # Stream Ollama response + speak sentence by sentence
        ask_and_speak(user_input)
