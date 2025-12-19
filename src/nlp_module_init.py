from transformers import pipeline

def load_nlp_model():
    """Carga del modelo ROBERTa sugerido para análisis de texto[cite: 24]."""
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")