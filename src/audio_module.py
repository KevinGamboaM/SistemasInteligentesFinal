import whisper
import torch
import logging

def transcribe_audio(video_path):
    """
    Usa Whisper para transcribir el audio del video.
    Devuelve una lista de diccionarios con start, end y text.
    """
    logging.info("Cargando modelo Whisper (puede tardar un poco)...")
    
    # Usa GPU si está disponible (recomendado para Colab), sino CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cargamos el modelo 'base' que es rápido y preciso
    model = whisper.load_model("base", device=device)
    
    logging.info(f"Transcribiendo: {video_path}")
    result = model.transcribe(video_path)
    
    segments_clean = []
    for segment in result["segments"]:
        segments_clean.append({
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"].strip()
        })
        # Imprimimos para ver progreso
        print(f"   🗣 [{round(segment['start'], 2)}s]: {segment['text'][:50]}...")
        
    return segments_clean