import speech_recognition as sr
import pyttsx3
import requests
import json
import re

# --- 1. TTS Setup (Speaking) ---
def speak(text):
    print(f"Bot: {text}")
    try:
        # Re-initializing the engine every time fixes a known Windows loop bug
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        else:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 170) 
        engine.setProperty('volume', 1.0) 
        
        engine.say(text)
        engine.runAndWait()
        engine.stop() # Force the event loop to close cleanly
    except Exception as e:
        print(f"Text-to-Speech Error: {e}")

# --- 2. Ollama Setup (Thinking) ---
def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3:8b",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "Sorry, I couldn't reach my brain."
    except Exception as e:
        return "Make sure Ollama is running in the background."

# --- 3. STT Setup (Listening) ---
recognizer = sr.Recognizer()

# Print out available microphones for debugging
print("\n--- Available Microphones ---")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"[{index}] {name}")
print("-----------------------------\n")

def listen():
    with sr.Microphone() as source:
        print("\nAdjusting to background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1.5) # Increased duration
        print("Listening... (Speak loudly into your laptop mic)")
        try:
            # Increased timeout so it waits longer for you to speak
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Processing text...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except Exception as e:
            print(f"Microphone error: {e}")
            return None

# --- Main Loop ---
if __name__ == "__main__":
    speak("Hello! I am online and ready to chat.")
    while True:
        user_input = listen()
        if user_input:
            if "stop" in user_input.lower() or "exit" in user_input.lower():
                speak("Goodbye!")
                break
            
            # Send the text to Ollama and get the response
            bot_response = ask_ollama(user_input)
            
            # Clean the text to prevent pyttsx3 from crashing on special characters
            clean_response = re.sub(r'```.*?```', ' [Code Example Provided on Screen] ', bot_response, flags=re.DOTALL)
            clean_response = re.sub(r'[*#_`]', '', clean_response)
            
            # Read the simplified response out loud
            speak(clean_response)
