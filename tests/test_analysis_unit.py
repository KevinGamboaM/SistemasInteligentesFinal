# tests/test_analysis_unit.py
from src.analysis import contar_emociones

def test_contar_emociones_basico():
    data = [
        {"emotion": "alegría"},
        {"emotion": "alegría"},
        {"emotion": "tristeza"}
    ]

    resultado = contar_emociones(data)

    assert resultado["alegría"] == 2
    assert resultado["tristeza"] == 1
