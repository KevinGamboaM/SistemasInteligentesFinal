import whisper # [cite: 24, 39]

def test_transcription(audio_path):
    """Prueba básica de transcripción de audio con Whisper[cite: 24, 39]."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    print("Texto detectado:", result["text"])
    return result["text"], result["segments"]