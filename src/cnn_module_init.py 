import cv2
import os

def extract_frames(video_path, output_folder, interval=1):
    """
    Script de extracción de frames (Día 1).
    Extrae una imagen cada 'n' segundos para análisis posterior.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Guardar frame según el intervalo de tiempo [cite: 24]
        if count % int(fps * interval) == 0:
            name = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(name, frame)
        count += 1
    cap.release()