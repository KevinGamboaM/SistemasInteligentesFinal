import os
import logging
from src.utils import ensure_directories, save_analysis_json
# Importamos los módulos de los chicos
from src.cnn_module import analyze_video_emotions
from src.audio_module import transcribe_audio
from src.nlp_module import analyze_sentiment_roberta

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    print("--- 🚀 INICIANDO EJECUCIÓN DEL DÍA 2 ---")
    ensure_directories()
    
    # 1. VIDEO DE PRUEBA 
    video_filename = "video_validacion_01.mp4"
    video_path = os.path.join("data", "raw", video_filename)
    
    if not os.path.exists(video_path):
        logging.error(f"No se encontró el video: {video_path}")
        print("⚠️  URGENTE: Graben un video corto y guárdenlo en data/raw/video_validacion_01.mp4")
        return

    # 2. EJECUCIÓN DE VISIÓN (Dilan)
    print("\n[1/3] 👁️  Procesando emociones faciales (Dilan)...")
    vision_data = analyze_video_emotions(video_path, sample_rate=1) # 1 frame cada segundo

    # 3. EJECUCIÓN DE AUDIO (Nico)
    print("\n[2/3] 🎤 Transcribiendo audio (Nico)...")
    audio_segments = transcribe_audio(video_path)

    # 4. EJECUCIÓN DE TEXTO (Eduardo)
    print("\n[3/3] 🧠 Analizando sentimiento del texto (Eduardo)...")
    # Enriquecemos los segmentos de audio con el análisis de texto
    for segment in audio_segments:
        emocion_texto = analyze_sentiment_roberta(segment['text'])
        segment['analisis_texto'] = emocion_texto

    # 5. GUARDADO DE RESULTADOS (Integración)
    print("\n💾 Guardando resultados integrados...")
    final_output = {
        "archivo": video_filename,
        "vision_artificial": vision_data,
        "audio_y_nlp": audio_segments
    }
    
    save_analysis_json(final_output, f"resultado_dia2_{video_filename}.json")
    print("✅ ¡Pipeline finalizado! Revisen la carpeta data/processed/")

if __name__ == "__main__":
    run_pipeline()