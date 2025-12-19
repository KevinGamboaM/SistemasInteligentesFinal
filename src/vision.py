
import cv2
from deepface import DeepFace
import pandas as pd
import json
from tqdm import tqdm

def procesar_emociones_video(video_path, output_json_path, sample_rate=5):
    print(f"👁️ [VISIÓN] Procesando video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return pd.DataFrame() # Retorna vacío si falla

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    datos_emociones = []

    for frame_num in tqdm(range(0, total_frames, sample_rate), desc="Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret: break

        timestamp = frame_num / fps
        try:
            result = DeepFace.analyze(frame, actions=['emotion'],
                                    enforce_detection=False,
                                    detector_backend='opencv',
                                    silent=True)
            face_data = result[0]
            
            registro = {
                "segundo": round(float(timestamp), 2),
                "emocion_dominante": str(face_data['dominant_emotion']),
                "confianza": float(face_data['face_confidence'])
            }
            datos_emociones.append(registro)
        except:
            pass

    cap.release()

    # Guardar JSON
    with open(output_json_path, 'w') as f:
        json.dump(datos_emociones, f, indent=4)
        
    return pd.DataFrame(datos_emociones)