import asyncio

import numpy as np
from faster_whisper import WhisperModel


class WhisperEngine:
    """Wrapper assíncrono do faster_whisper.WhisperModel.

    O modelo é carregado uma única vez e mantido em memória durante toda a sessão.
    A transcrição roda em asyncio.to_thread para nunca bloquear o event loop.
    Como um único worker alimenta o modelo sequencialmente, não é necessário lock.

    vad_filter é desativado intencionalmente: o VAD upstream (SileroVAD) já garante que apenas áudio com fala chega aqui.

    Parâmetros
    ----------
    model_size : str
        Tamanho do modelo Whisper ('tiny', 'base', 'small',
        'medium', 'large-v3').
    device : str
        'cpu' para Raspberry Pi; 'cuda' se houver GPU disponível.
    compute_type : str
        Estratégia de quantização. 'int8' equilibra velocidade e qualidade na CPU.
    language : str
        Código BCP-47 do idioma (ex.: 'pt' para português).
    beam_size : int
        Número de beams na busca. Valores menores reduzem latência.
    initial_prompt : str | None
        Texto inicial que contextualiza o domínio e reduz erros de vocabulário.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "pt",
        beam_size: int = 3,
        initial_prompt: str | None = (
            "Olá. Sou um tutor de ensino fundamental virtual. Como posso te ajudar hoje?"
        ),
    ):
        self.language = language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

        print(f"Carregando Whisper '{model_size}' ({device}, {compute_type})...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper pronto.")

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:

        return await asyncio.to_thread(self._transcribe_sync, pcm_bytes, sample_rate)

    def _transcribe_sync(self, pcm_bytes: bytes, sample_rate: int) -> str:

        # Converte bytes brutos para float32 normalizado sem arquivo temporário
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,  # VAD já aplicado upstream pelo SileroVAD
            initial_prompt=self.initial_prompt,
            log_prob_threshold=-0.85,  # descarta palavras com baixa confiança
            no_speech_threshold=0.6,  # ignora segmentos com >= 60 % de silêncio
        )

        return " ".join(s.text for s in segments).strip()
