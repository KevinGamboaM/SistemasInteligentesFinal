# tests/test_integration.py
import json
import os

from src.analysis import analizar_vision_y_audio
from src.report import generar_reporte

def test_pipeline_completo(tmp_path):
    vision_data = [
        {"emotion": "alegría"},
        {"emotion": "tristeza"}
    ]

    audio_data = [
        {"emocion": "alegría"},
        {"emocion": "alegría"}
    ]

    vision_file = tmp_path / "vision.json"
    audio_file = tmp_path / "audio.json"
    report_file = tmp_path / "reporte.txt"

    json.dump(vision_data, open(vision_file, "w"))
    json.dump(audio_data, open(audio_file, "w"))

    resultado = analizar_vision_y_audio(
        str(vision_file),
        str(audio_file)
    )

    generar_reporte(resultado, str(report_file))

    assert report_file.exists()

    contenido = report_file.read_text()
    assert "alegría" in contenido
