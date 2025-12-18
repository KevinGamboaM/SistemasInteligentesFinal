import os
import logging
# ... (imports anteriores de utils, cnn, audio, nlp, sync) ...
from src.utils import ensure_directories, save_analysis_json
from src.cnn_module import analyze_video_emotions
from src.audio_module import transcribe_audio
from src.nlp_module import analyze_sentiment_roberta
from src.sync_module import merge_multimodal_data
from src.temporal_module import predict_interview_score # NUEVO IMPORT

def run_pipeline():
    print("--- 🚀 DÍA 4: SISTEMA COMPLETO CON IA ---")
    ensure_directories()
    
    video_filename = "video_validacion_01.mp4" # Asegúrense de tener este video
    video_path = os.path.join("data", "raw", video_filename)
    
    if not os.path.exists(video_path):
        print("❌ Falta el video.")
        return

    # [PASOS 1, 2 y 3 IGUALES A AYER...]
    print("\n1. Procesando Video y Audio...")
    vision = analyze_video_emotions(video_path)
    audio = transcribe_audio(video_path)
    
    print("\n2. Analizando Texto...")
    for seg in audio:
        seg['analisis_texto'] = analyze_sentiment_roberta(seg['text'])
        
    print("\n3. Sincronizando...")
    data_integrada = merge_multimodal_data(vision, audio)

    # [PASO 4: NUEVO - PREDICCIÓN CON LSTM]
    print("\n4. 🧠 Consultando al Modelo Temporal (RNN)...")
    score_congruencia = predict_interview_score("models/interview_lstm.pth", data_integrada)
    
    veredicto = "CONGRUENTE" if score_congruencia > 0.6 else "POSIBLE INCONGRUENCIA"
    
    final_report = {
        "video": video_filename,
        "score_ia": score_congruencia,
        "veredicto_sistema": veredicto,
        "detalle_secuencia": data_integrada
    }
    
    save_analysis_json(final_report, f"reporte_final_dia4.json")
    print(f"\n📊 SCORE FINAL: {score_congruencia*100:.1f}% ({veredicto})")
    print("✅ Sistema Inteligente finalizado.")

if __name__ == "__main__":
    run_pipeline()