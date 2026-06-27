import asyncio
from unittest.mock import MagicMock, patch

import numpy as np


@patch("stt.whisper_engine.WhisperModel")
def test_transcribe_retorna_texto(mock_whisper):
    seg1, seg2 = MagicMock(), MagicMock()
    seg1.text = "quanto"
    seg2.text = "é dois mais dois"

    mock_whisper.return_value.transcribe.return_value = (
        [seg1, seg2],
        MagicMock(),
    )

    from stt.whisper_engine import WhisperEngine

    engine = WhisperEngine(model_size="tiny")
    pcm = np.zeros(16000, dtype=np.int16).tobytes()

    result = asyncio.run(engine.transcribe(pcm))

    assert result == "quanto é dois mais dois"


@patch("stt.whisper_engine.WhisperModel")
def test_transcribe_vazio(mock_whisper):
    mock_whisper.return_value.transcribe.return_value = (
        [],
        MagicMock(),
    )

    from stt.whisper_engine import WhisperEngine

    engine = WhisperEngine(model_size="tiny")
    pcm = np.zeros(16000, dtype=np.int16).tobytes()

    result = asyncio.run(engine.transcribe(pcm))

    assert result == ""
