import whisper
from transformers import pipeline
import json
import re

print("⏳ [AUDIO] Cargando modelos...")
modelo_whisper = whisper.load_model("base")
analizador_emocion = pipeline("text-classification", model="pysentimiento/robertuito-emotion-analysis")

def analizar_emocion_texto(texto):
    if not texto or len(texto) < 2: return "neutral"
    try:
        res = analizador_emocion(texto[:120])[0]
        return res['label']
    except:
        return "neutral"

def subdividir_segmento(segmento, max_palabras=4):
    texto_completo = segmento["text"].strip()
    start_time = segmento["start"]
    end_time = segmento["end"]
    duration = end_time - start_time
    
    frases = texto_completo.split() # Simplificado para demo
    if not frases: return []

    sub_segmentos = []
    current_time = start_time
    chunk_duration = duration / max(1, (len(frases) / max_palabras))

    # Lógica simplificada de troceado
    chunks = [" ".join(frases[i:i+max_palabras]) for i in range(0, len(frases), max_palabras)]
    
    for chunk in chunks:
        emocion = analizar_emocion_texto(chunk)
        sub_segmentos.append({
            "start": round(current_time, 2),
            "end": round(current_time + chunk_duration, 2),
            "texto": chunk,
            "emocion_texto": emocion
        })
        current_time += chunk_duration
        
    return sub_segmentos

def procesar_audio_detallado(video_path, output_path):
    print(f"🎤 [AUDIO] Transcribiendo...")
    resultado = modelo_whisper.transcribe(video_path, language="es")
    salida_final = []

    for seg in resultado["segments"]:
        trozos = subdividir_segmento(seg)
        salida_final.extend(trozos)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(salida_final, f, indent=4, ensure_ascii=False)
        
    return salida_final