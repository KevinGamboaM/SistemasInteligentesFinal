import cv2
from deepface import DeepFace
import pandas as pd
import json
from tqdm import tqdm

def procesar_emociones_video(video_path, output_json_path, sample_rate=5):
    """
    Versión corregida que convierte tipos NumPy a Python nativo para evitar error de JSON.
    """

    # 1. Cargar video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    datos_emociones = []

    print(f"Iniciando análisis de {total_frames} frames...")

    # 2. Iterar sobre el video
    for frame_num in tqdm(range(0, total_frames, sample_rate)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_num / fps

        try:
            # 3. Aplicar DeepFace
            result = DeepFace.analyze(frame, actions=['emotion'],
                                    enforce_detection=False,
                                    detector_backend='opencv',
                                    silent=True)

            face_data = result[0]

            # --- CORRECCIÓN AQUÍ ---
            # Convertimos todo a tipos nativos de Python (float, int, str)
            registro = {
                "segundo": round(float(timestamp), 2),  # Convertir a float normal
                "emocion_dominante": str(face_data['dominant_emotion']),
                "confianza": float(face_data['face_confidence']), # Convertir numpy.float32 a float
                # Convertir el diccionario de emociones una por una
                "detalles": {k: float(v) for k, v in face_data['emotion'].items()}
            }
            # -----------------------

            datos_emociones.append(registro)

        except Exception as e:
            pass

    cap.release()

    # 4. Guardar resultados
    try:
        with open(output_json_path, 'w') as f:
            json.dump(datos_emociones, f, indent=4)
        print(f"✅ ¡ÉXITO! Datos guardados en: {output_json_path}")
    except Exception as e:
        print(f"Error al guardar JSON: {e}")

    # Mostrar vista previa
    df = pd.DataFrame(datos_emociones)
    return df

# --- ZONA DE EJECUCIÓN ---
# Asegúrate de que esta variable tenga la ruta correcta (la que copiaste antes)
nombre_video = "/content/example1.mp4"  # <--- REVISA QUE ESTO SIGA BIEN
nombre_salida = "datos_vision.json"

import os
if os.path.exists(nombre_video):
    df_resultado = procesar_emociones_video(nombre_video, nombre_salida)
    # Si todo sale bien, verás una tabla abajo
    if df_resultado is not None and not df_resultado.empty:
        print("\nVista previa de los primeros 5 registros:")
        print(df_resultado.head())
else:
    print(f"⚠️ NO SE ENCUENTRA EL ARCHIVO: {nombre_video}")
    print("Por favor verifica la ruta (Click derecho en el archivo -> Copiar ruta)")