import numpy as np

def merge_multimodal_data(vision_data, audio_data):
    """
    Sincroniza los datos de visión y audio basándose en timestamps.
    """
    merged_timeline = []

    print(f"🔄 Sincronizando {len(vision_data)} frames con {len(audio_data)} segmentos de audio...")

    for segment in audio_data:
        start_time = segment['start']
        end_time = segment['end']
        
        # 1. Filtrar frames que ocurrieron mientras se decía esta frase
        frames_in_segment = [
            f for f in vision_data 
            if start_time <= f['timestamp'] <= end_time
        ]

        # 2. Calcular la emoción facial predominante en este lapso
        if frames_in_segment:
            # Extraemos todas las emociones detectadas en este segmento
            emotions = [f['emocion'] for f in frames_in_segment]
            # Buscamos la más frecuente (moda)
            dominant_face_emotion = max(set(emotions), key=emotions.count)
        else:
            dominant_face_emotion = "neutral" # Fallback si no hay frames

        # 3. Determinar congruencia (Lógica simple para Día 3)
        text_emotion = segment.get('analisis_texto', {}).get('emocion_texto', 'neutral')
        
        # Regla simple: Si son iguales o compatibles, es congruente
        # (Se puede mejorar mañana para el RNN)
        es_congruente = (dominant_face_emotion == text_emotion)

        # 4. Crear el objeto fusionado
        merged_entry = {
            "timestamp_start": start_time,
            "timestamp_end": end_time,
            "texto": segment['text'],
            "emocion_texto": text_emotion,
            "emocion_rostro": dominant_face_emotion,
            "congruencia": es_congruente
        }
        
        merged_timeline.append(merged_entry)

    return merged_timeline