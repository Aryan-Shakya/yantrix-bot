from piper.voice import PiperVoice
import wave, os, struct

MODEL = r"C:\Users\Aryan Shakya\Desktop\BOT\en_US-lessac-medium.onnx"
OUT   = r"C:\Users\Aryan Shakya\Desktop\BOT\piper_test_output.wav"

try:
    print("[1] Loading Piper voice...")
    voice = PiperVoice.load(MODEL)
    print(f"[2] Sample rate: {voice.config.sample_rate}")

    print("[3] Trying synthesize_stream_raw()...")
    raw_chunks = list(voice.synthesize_stream_raw("Hello, this is a Piper TTS test."))
    total_raw = b"".join(raw_chunks)
    print(f"[4] Got {len(total_raw)} raw PCM bytes")

    if len(total_raw) > 0:
        with wave.open(OUT, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            wf.writeframes(total_raw)
        print(f"[5] WAV saved: {os.path.getsize(OUT)} bytes -> {OUT}")
    else:
        print("[5] FAILED: synthesize_stream_raw returned no bytes!")

    print("[6] Trying synthesize() with wave file...")
    with wave.open(OUT + "2.wav", "wb") as wf2:
        wf2.setnchannels(1)
        wf2.setsampwidth(2)
        wf2.setframerate(voice.config.sample_rate)
        voice.synthesize("Hello world test.", wf2)
    print(f"[7] synthesize() WAV size: {os.path.getsize(OUT + '2.wav')} bytes")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("DONE")
