import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importamos los módulos de la carpeta src
from src.vision import procesar_emociones_video
from src.audio import procesar_audio_detallado
from src.ia_model import predecir_con_ia

# CONFIGURACIÓN
VIDEO_FILE = "video_validacion_01.mp4" # <--- ¡PON TU VIDEO EN data/raw/ Y CAMBIA ESTO!
RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
VIDEO_PATH = os.path.join(RAW_DIR, VIDEO_FILE)

# Mapeos de traducción
TRADUCCION_FACE = {
    "happy": "joy", "sad": "sadness", "angry": "anger",
    "surprise": "surprise", "fear": "fear", "neutral": "neutral"
}

def sincronizar(df_v, data_a):
    print("🔄 Sincronizando datos...")
    df_a = pd.DataFrame(data_a)
    df_integrado = df_v.copy()
    df_integrado['emocion_texto'] = "neutral"
    
    for idx, row in df_integrado.iterrows():
        t = row['segundo']
        match = df_a[(df_a['start'] <= t) & (df_a['end'] >= t)]
        if not match.empty:
            df_integrado.at[idx, 'emocion_texto'] = match.iloc[0]['emocion_texto']
            
    # Traducir columna facial para que coincida con texto
    df_integrado['emocion_facial'] = df_integrado['emocion_dominante'].map(TRADUCCION_FACE).fillna("neutral")
    
    return df_integrado

def graficar(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='segundo', y='emocion_facial', color='blue', label='Cara', s=50)
    sns.scatterplot(data=df, x='segundo', y='emocion_texto', color='red', label='Texto', s=50, marker="x")
    plt.title("Análisis Multimodal: Cara vs Texto")
    plt.xlabel("Tiempo (s)")
    plt.grid(True, alpha=0.3)
    plt.savefig("grafico_final.png")
    print("📈 Gráfico guardado como 'grafico_final.png'")

def main():
    if not os.path.exists(RAW_DIR): os.makedirs(RAW_DIR)
    if not os.path.exists(PROC_DIR): os.makedirs(PROC_DIR)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: No encuentro el video en {VIDEO_PATH}")
        print("Por favor crea la carpeta data/raw y pon tu video ahí.")
        return

    # 1. Visión
    df_vision = procesar_emociones_video(VIDEO_PATH, os.path.join(PROC_DIR, "vision.json"))
    
    # 2. Audio
    data_audio = procesar_audio_detallado(VIDEO_PATH, os.path.join(PROC_DIR, "audio.json"))
    
    # 3. Sincronización
    df_final = sincronizar(df_vision, data_audio)
    
    # 4. Gráficos (Tu código original)
    graficar(df_final)
    
    # 5. IA (Mi código nuevo)
    score = predecir_con_ia(df_final)
    
    print("\n" + "="*40)
    print(f"🤖 VEREDICTO FINAL IA")
    print(f"Probabilidad de Congruencia: {score*100:.1f}%")
    if score > 0.6:
        print("✅ EL CANDIDATO ES CONGRUENTE")
    else:
        print("⚠️ ALERTA: POSIBLE INCONGRUENCIA DETECTADA")
    print("="*40)

if __name__ == "__main__":
    main()