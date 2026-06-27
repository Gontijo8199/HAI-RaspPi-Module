import torch
import numpy as np
from typing import Tuple


class SileroVAD:
    """Wrapper do Silero VAD para classificação frame a frame.

    Exige áudio mono PCM-16 a 16 kHz em chunks de exatamente 512 amostras
    (32 ms), restrição imposta pelo modelo.

    Parâmetros
    ----------
    threshold : float
        Probabilidade mínima para classificar um frame como fala.
        Valores menores aumentam sensibilidade; maiores reduzem falsos positivos.
    sample_rate : int
        Deve ser 16000. Qualquer outro valor levanta ValueError.
    """

    CHUNK_SAMPLES = 512  # 32 ms a 16 kHz (exigido pelo Silero)

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        if sample_rate != 16000:
            raise ValueError("SileroVAD suporta apenas 16000 Hz.")

        self.threshold = threshold
        self.sample_rate = sample_rate

        print("Carregando modelo Silero VAD...")
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        self._model.eval()
        print("Silero VAD pronto.")

    def reset_state(self) -> None:
        self._model.reset_states()

    def is_speech(self, pcm_chunk: bytes) -> Tuple[bool, float]:
        """Classifica um único chunk PCM-16 de 512 amostras.

        Parâmetros
        ----------
        pcm_chunk : bytes
            1024 bytes de áudio PCM-16 LE mono (512 amostras * 2 bytes).

        Retorna
        -------
        is_speech : bool
            True se a probabilidade excede threshold.
        prob : float
            Probabilidade bruta de fala estimada pelo modelo.
        """
        audio = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, 512)

        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()

        return prob >= self.threshold, prob