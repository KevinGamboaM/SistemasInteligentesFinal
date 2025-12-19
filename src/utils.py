import json

def format_output(timestamp, face_emotion, text_emotion):
    """Define la interfaz entre módulos definida en la reunión del Día 1[cite: 43]."""
    return {
        "timestamp": timestamp,
        "emocion_facial": face_emotion,
        "emocion_texto": text_emotion,
        "congruencia": face_emotion == text_emotion
    }

def normalize_emotion(emotion_raw):
    """
    Normaliza las emociones de diferentes librerías a un estándar común.
    DeepFace: angry, disgust, fear, happy, sad, surprise, neutral
    Roberta: joy, optimism, anger, sadness, fear, surprise
    """
    emotion = emotion_raw.lower()
    
    mapping = {
        "joy": "happy",
        "optimism": "happy",
        "happiness": "happy",
        "anger": "angry",
        "sadness": "sad",
        "disgust": "angry", # Simplificación para el examen
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral"
    }  
    return mapping.get(emotion, "neutral")