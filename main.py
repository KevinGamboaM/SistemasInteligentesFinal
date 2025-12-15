# main.py
from src.utils import ensure_directories
from src.cnn_module import extract_frames
from src.audio_module import test_transcription

def main():
    print("🚀 Iniciando Pipeline de Análisis de Entrevistas...")
    
    # 1. Asegurar carpetas
    ensure_directories()
    
    # 2. Ruta del video de prueba (debe estar en data/raw)
    video_path = "data/raw/entrevista_prueba.mp4"
    
    # Nota: Aquí irán las llamadas a los módulos integrados en el Día 3
    print("Estado: Entorno y módulos base listos.")

if __name__ == "__main__":
    main()