import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def settings():
    return {
        "api": {"model": "gemma-4-26b-a4b-it"},
        "stt": {
            "language": "pt",
            "whisper_model": "tiny",  # modelo leve para CI
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
        },
        "vad": {
            "threshold": 0.5,
            "preroll_ms": 500,
            "silence_ms": 750,
        },
    }


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.send.return_value = "Resposta de teste."
    return client
