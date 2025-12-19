# tests/test_vision_unit.py
import pytest
from unittest.mock import MagicMock, patch
import json
import os

from src.vision import procesar_emociones_video

@patch("src.vision.cv2.VideoCapture")
@patch("src.vision.DeepFace.analyze")
def test_procesar_emociones_video_genera_json(mock_analyze, mock_video):
    # Mock del video
    fake_video = MagicMock()
    fake_video.read.side_effect = [
        (True, "frame1"),
        (False, None)
    ]
    fake_video.get.return_value = 30
    mock_video.return_value = fake_video

    # Mock DeepFace
    mock_analyze.return_value = [{
        "dominant_emotion": "happy",
        "emotion": {"happy": 90, "sad": 10}
    }]

    output_path = "test_output.json"

    procesar_emociones_video("fake.mp4", output_path, sample_rate=1)

    assert os.path.exists(output_path)

    with open(output_path) as f:
        data = json.load(f)

    assert len(data) == 1
    assert data[0]["emotion"] == "happy"

    os.remove(output_path)
