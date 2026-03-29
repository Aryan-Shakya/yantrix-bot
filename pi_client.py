import speech_recognition as sr
import requests
import json
import re
import os
import subprocess

# ══════════════════════════════════════════════════════════════════
# YANTRIX BOT — Raspberry Pi Client
# Handles: mic input → STT (laptop) → RAG context → Ollama LLM → TTS (laptop) → speaker
# ══════════════════════════════════════════════════════════════════

LAPTOP_IP   = "10.101.196.138"          # ← update to your laptop's Wi-Fi IP

OLLAMA_URL  = f"http://{LAPTOP_IP}:11434/api/chat"
WHISPER_URL = f"http://{LAPTOP_IP}:8000/api/stt"
TTS_URL     = f"http://{LAPTOP_IP}:8000/api/tts"
FILLER_URL  = f"http://{LAPTOP_IP}:8000/api/filler"

FILLER_FILE = "/home/yantrix/filler_audio.wav"

# ── System prompt — defines bot persona and RAG injection format
SYSTEM_PROMPT = """You are Yantrix, the official AI reception assistant for DY Patil International University (DYPIU), Akurdi, Pune.

Your job is to help students, parents, and visitors by answering questions about the university accurately and concisely.

Rules you must follow:
- Always answer in 2-3 short, clear sentences. Never ramble.
- If college information is provided in the context block below, use it to answer. Prioritise it over your general knowledge.
- If the context does not contain the answer, use your general knowledge about DYPIU and universities.
- Never say "I don't know" — always try to give a helpful response or direct them to the admissions office at +91 9071123434.
- Do not use bullet points, markdown, or any special formatting. Speak naturally.
- Be warm, professional, and concise — like a real reception assistant."""

# Conversation memory (keeps last 5 exchanges for context)
memory = [{"role": "system", "content": SYSTEM_PROMPT}]


# ══════════════════════════════════════════════════════════════════
# 1. FILLER AUDIO — download once at startup for instant playback
# ══════════════════════════════════════════════════════════════════
def download_filler():
    try:
        r = requests.get(FILLER_URL, timeout=10)
        with open(FILLER_FILE, "wb") as f:
            f.write(r.content)
        print(f"   -> Filler audio ready ({os.path.getsize(FILLER_FILE)} bytes)")
    except Exception as e:
        print(f"   -> Could not download filler: {e}")

def play_filler():
    """Non-blocking filler sound while Ollama thinks."""
    if os.path.exists(FILLER_FILE):
        subprocess.Popen(
            ["aplay", FILLER_FILE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


# ══════════════════════════════════════════════════════════════════
# 2. TTS — send text to Piper on laptop, play audio on Pi speaker
# ══════════════════════════════════════════════════════════════════
def speak(text):
    print(f"\nBot: {text}")
    try:
        response = requests.post(TTS_URL, json={"text": text}, timeout=60)
        if response.status_code == 200:
            filename = "/tmp/response_audio.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            os.system(f"aplay {filename} 2>/dev/null")
            os.remove(filename)
        else:
            print(f"   [TTS ERROR] Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   [TTS ERROR] Cannot reach laptop!")
    except requests.exceptions.Timeout:
        print("   [TTS ERROR] TTS timed out.")
    except Exception as e:
        print(f"   [TTS ERROR] {e}")


# ══════════════════════════════════════════════════════════════════
# 3. LLM — send RAG-enriched prompt to Ollama, stream response
#    Speaks sentence-by-sentence for low latency
# ══════════════════════════════════════════════════════════════════
def clean_text(text):
    """Strip markdown symbols that sound bad when spoken aloud."""
    text = re.sub(r'```.*?```', '[code example]', text, flags=re.DOTALL)
    text = re.sub(r'[*#_`]', '', text)
    return text.strip()

def ask_and_speak(user_query, context=""):
    global memory

    # ── Build RAG-enriched user message
    if context:
        enriched = (
            f"Use the following verified information about DY Patil International University "
            f"to answer the question. Prioritise this over anything else.\n\n"
            f"--- College Knowledge Base ---\n{context}\n"
            f"------------------------------\n\n"
            f"Visitor question: {user_query}"
        )
    else:
        enriched = user_query

    memory.append({"role": "user", "content": enriched})

    payload = {
        "model": "llama3.2:3b",
        "messages": memory,
        "stream": True,
        "keep_alive": -1
    }

    try:
        print("   [Brain] Streaming from Ollama...")
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=30)

        full_reply    = ""
        sentence_buf  = ""

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            sentence_buf += token
            full_reply   += token

            # Speak as soon as a complete sentence is ready
            if sentence_buf.rstrip().endswith((".", "!", "?", "...", ":")):
                sentence = clean_text(sentence_buf)
                sentence_buf = ""
                if sentence:
                    speak(sentence)

            if chunk.get("done"):
                break

        # Speak any leftover text
        if sentence_buf.strip():
            leftover = clean_text(sentence_buf)
            if leftover:
                speak(leftover)

        # Store assistant reply in memory (store original query for cleaner context)
        memory[-1] = {"role": "user", "content": user_query}
        memory.append({"role": "assistant", "content": full_reply})

        # Keep memory bounded: system prompt + last 10 messages
        if len(memory) > 11:
            memory = [memory[0]] + memory[-10:]

    except requests.exceptions.ConnectionError:
        speak("I cannot connect to my brain. Please check if Ollama is running on the laptop.")
    except Exception as e:
        speak("Sorry, something went wrong. Please try again.")
        print(f"   [Brain ERROR] {e}")


# ══════════════════════════════════════════════════════════════════
# 4. STT — record from Pi mic, send WAV to Whisper on laptop
#    Returns (transcript, rag_context)
# ══════════════════════════════════════════════════════════════════
recognizer = sr.Recognizer()
recognizer.pause_threshold     = 0.8
recognizer.non_speaking_duration = 0.5

def listen():
    with sr.Microphone() as source:
        print("\n[Listening] Adjusting for noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("[Listening] Speak now...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

            with open("/tmp/temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            with open("/tmp/temp_audio.wav", "rb") as f:
                r = requests.post(WHISPER_URL, files={"file": f}, timeout=20)

            result  = r.json()
            text    = result.get("text", "").strip()
            context = result.get("context", "")

            if text:
                print(f"You said: {text}")
                if context:
                    print(f"[RAG] Context retrieved ({len(context)} chars)")
                else:
                    print("[RAG] No relevant context found — using general knowledge")
                return text, context
            else:
                print("[STT] Empty transcript — speak louder or closer to the mic")
                return None, None

        except sr.WaitTimeoutError:
            print("[STT] No speech detected.")
            return None, None
        except requests.exceptions.ConnectionError:
            print(f"[STT ERROR] Cannot reach laptop at {LAPTOP_IP}:8000")
            return None, None
        except requests.exceptions.Timeout:
            print("[STT ERROR] Whisper took too long.")
            return None, None
        except Exception as e:
            print(f"[STT ERROR] {e}")
            return None, None


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("==================================================")
    print("YANTRIX BOT — DY Patil International University")
    print("Downloading filler audio from laptop server...")
    download_filler()

    speak("Hello! I am Yantrix, the reception assistant for DY Patil International University. How can I help you today?")

    while True:
        user_input, context = listen()

        if not user_input:
            continue

        if any(w in user_input.lower() for w in ["stop", "exit", "shutdown", "bye", "goodbye"]):
            speak("Thank you for visiting DY Patil International University. Have a great day!")
            break

        # Play filler instantly while RAG + Ollama work in background
        play_filler()

        # Generate RAG-enriched response and speak it
        ask_and_speak(user_input, context)
