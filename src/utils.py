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