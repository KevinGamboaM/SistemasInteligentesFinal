import torch
import torch.nn as nn
import numpy as np

# Mapeo simple de emociones a números para que la red entienda
EMOTION_MAP = {
    "happy": 1.0, "joy": 1.0, "optimism": 1.0,
    "neutral": 0.5,
    "sad": 0.0, "sadness": 0.0,
    "angry": -0.5, "anger": -0.5, "disgust": -0.5,
    "fear": -1.0, "surprise": 0.2
}

def text_to_vector(face_emo, text_emo):
    """Convierte palabras a vector numérico [Face_Val, Text_Val]."""
    v1 = EMOTION_MAP.get(face_emo, 0.5)
    v2 = EMOTION_MAP.get(text_emo, 0.5)
    return [v1, v2]

class InterviewLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, num_layers=1):
        super(InterviewLSTM, self).__init__()
        # LSTM: Memoria a corto y largo plazo
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Capa final para decidir (0 a 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Tomamos el último estado de la secuencia (resumen final)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Función Helper para usar el modelo ya entrenado
def predict_interview_score(model_path, segments_json):
    model = InterviewLSTM()
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except:
        return 0.5 # Retorno neutro si no hay modelo entrenado aún

    # Preprocesar datos reales para el modelo
    input_seq = []
    for seg in segments_json:
        vec = text_to_vector(seg.get('emocion_rostro', 'neutral'), 
                             seg.get('emocion_texto', 'neutral'))
        input_seq.append(vec)
    
    if not input_seq: return 0.0

    input_tensor = torch.tensor([input_seq], dtype=torch.float32)
    with torch.no_grad():
        score = model(input_tensor).item()
    
    return round(score, 4)