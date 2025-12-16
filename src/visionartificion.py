import cv2
from deepface import DeepFace
import logging

def analyze_video_emotions(video_path, sample_rate=1):
    """
    Recorre el video y detecta emociones cada 'sample_rate' segundos.
    """
    results = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logging.error(f"Error al abrir video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    logging.info(f"Iniciando análisis de DeepFace en: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Procesamos solo si es el momento correcto (ej: cada segundo)
        if frame_count % int(fps * sample_rate) == 0:
            timestamp = frame_count / fps
            try:
                # DeepFace hace la magia aquí
                objs = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv',
                    silent=True
                )
                
                # Extraemos la emoción dominante
                dominant = objs[0]['dominant_emotion']
                results.append({
                    "timestamp": round(timestamp, 2),
                    "emocion": dominant
                })
                print(f"   ⏱ {round(timestamp, 2)}s: {dominant}")
            
            except Exception as e:
                # Si falla (ej. cara tapada), lo ignoramos o ponemos error
                pass
        
        frame_count += 1

    cap.release()
    return results