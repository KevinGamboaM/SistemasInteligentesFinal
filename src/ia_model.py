import numpy as np
import pandas as pd
import tensorflow as tf  # <--- AQUÍ ESTABA EL ERROR (antes decía 'as pd')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
INPUT_FILE = "datos_integrados_finales.csv"
WINDOW_SIZE = 5  # Cuántos segundos hacia atrás mira el modelo
EPOCHS = 20      # Iteraciones de entrenamiento

def preparar_datos_lstm(df):
    """
    Convierte las emociones de texto (ej: 'happy') a números y crea
    ventanas de tiempo para alimentar la LSTM.
    """
    # 1. Codificar emociones a números
    le = LabelEncoder()
    # Aseguramos que los datos sean strings para evitar errores
    df['emocion_facial'] = df['emocion_facial'].astype(str)
    df['emocion_facial_code'] = le.fit_transform(df['emocion_facial'])
    
    data = df['emocion_facial_code'].values
    
    # 2. Crear secuencias (X) y objetivos (y)
    X, y = [], []
    # Necesitamos al menos WINDOW_SIZE datos para empezar
    if len(data) <= WINDOW_SIZE:
        raise ValueError("El video es muy corto para el análisis de series de tiempo.")

    for i in range(len(data) - WINDOW_SIZE):
        X.append(data[i:i+WINDOW_SIZE])
        y.append(data[i+WINDOW_SIZE])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape para LSTM: [muestras, pasos_tiempo, características]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, le

def construir_modelo_lstm(input_shape, n_classes):
    """
    Define la arquitectura de la Red Neuronal Recurrente.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- EJECUCIÓN PRINCIPAL ---

# Verificamos si existe el archivo
if os.path.exists(INPUT_FILE):
    try:
        # 1. Cargar datos
        df = pd.read_csv(INPUT_FILE)
        print(f"✅ Datos cargados: {len(df)} registros.")
        
        # 2. Preprocesar
        X, y, encoder = preparar_datos_lstm(df)
        n_clases = len(encoder.classes_)
        
        # Separar Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"🧠 Entrenando LSTM en {len(X_train)} secuencias temporales...")
        
        # 3. Entrenar
        model = construir_modelo_lstm((WINDOW_SIZE, 1), n_clases)
        
        history = model.fit(X_train, y_train, 
                            epochs=EPOCHS, 
                            batch_size=4, 
                            validation_data=(X_test, y_test),
                            verbose=1)
        
        # 4. Evaluar
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print("\n" + "="*40)
        print(f"📊 REPORTE DE ESTABILIDAD EMOCIONAL (LSTM)")
        print("="*40)
        print(f"Precisión del Modelo Temporal: {accuracy*100:.2f}%")
        
        if accuracy > 0.7:
            conclusion = "ALTA. El sujeto es predecible y estable."
        elif accuracy > 0.4:
            conclusion = "MEDIA. Hay cambios emocionales frecuentes."
        else:
            conclusion = "BAJA. Comportamiento errático o muy cambiante."
            
        print(f"Estabilidad Detectada: {conclusion}")
        print("="*40)

        # 5. Gráfico
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Error Entrenamiento')
        plt.plot(history.history['val_loss'], label='Error Validación')
        plt.title('Entrenamiento Modelo LSTM (Series de Tiempo)')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        plt.savefig("grafico_lstm_loss.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error durante la ejecución del modelo: {e}")
else:
    print(f"❌ NO SE ENCUENTRA EL ARCHIVO: {INPUT_FILE}")
    print("Asegúrate de haber ejecutado el bloque de código anterior (Script 3) que genera el CSV.")