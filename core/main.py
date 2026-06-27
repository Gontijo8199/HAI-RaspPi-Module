# python main.py          VAD automático
# python main.py --ptt    modo Push-to-Talk via Enter

import sys
import tomllib
import re
import asyncio
import logging
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from stt.capture import WhisperStream
from stt.ptt import PttStream
from api.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

CONFIG_PATH  = Path(__file__).parent.parent / "config/settings.toml"
SECRETS_PATH = Path(__file__).parent.parent / "config/secrets.toml"

RESET_TRIGGERS = {"resetar", "reset", "nova sessão", "novo assunto", "recomeçar"}


def load_config():
    with open(CONFIG_PATH,  "rb") as f:
        settings = tomllib.load(f)
    with open(SECRETS_PATH, "rb") as f:
        secrets = tomllib.load(f)
    return settings, secrets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tutor Virtual HAI")
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="Ativa o modo Push-to-Talk (Enter para gravar/parar).",
    )
    return parser.parse_args()


async def processar_llm(client: LLMClient, utterance: str):
    if utterance.strip().lower() in RESET_TRIGGERS:
        client.resetar_sessao()
        print("\n[SESSÃO REINICIADA] Histórico apagado.")
        return

    print(f"\n[ENVIANDO PARA LLM] -> \"{utterance}\"")
    try:
        resposta = await asyncio.to_thread(client.send, utterance)
        print(f"\n[RESPOSTA] {resposta}")
    except Exception as e:
        print(f"\n[ERRO NA API] {e}")


async def async_main(ptt: bool):
    settings, secrets = load_config()

    api_key = secrets["api"]["key"]

    # [api]
    model = settings["api"]["model"]

    # [stt]
    lang          = settings["stt"]["language"]
    whisper_model = settings["stt"]["whisper_model"]
    whisper_device       = settings["stt"]["whisper_device"]
    whisper_compute_type = settings["stt"]["whisper_compute_type"]

    # [vad]
    vad_threshold = settings["vad"]["threshold"]
    preroll_ms    = settings["vad"]["preroll_ms"]
    silence_ms    = settings["vad"]["silence_ms"]

    if not api_key:
        raise ValueError("api.key não definida em config/secrets.toml")

    if ptt:
        stt = PttStream(
            language=lang,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
        )
    else:
        stt = WhisperStream(
            language=lang,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            vad_threshold=vad_threshold,
            preroll_ms=preroll_ms,
            silence_ms=silence_ms,
        )

    client = LLMClient(api_key=api_key, model=model)

    modo = "PTT (Enter)" if ptt else "VAD automático"
    print(f"Iniciando captura de voz [{modo}]... (Pressione Ctrl+C para sair)")
    await stt.start()

    try:
        while True:
            utterance = await stt.get_utterance()

            if not re.search(r'[a-zA-Z0-9á-úÁ-Ú]', utterance):
                print(f"Ignorando ruído/transcrição vazia: '{utterance}'")
                continue

            asyncio.create_task(processar_llm(client, utterance))
    except asyncio.CancelledError:
        print("\nDesligando microfone...")
    finally:
        stt.stop()


def main():
    args = parse_args()
    try:
        asyncio.run(async_main(ptt=args.ptt))
    except KeyboardInterrupt:
        print("\nEncerrando o programa.")


if __name__ == "__main__":
    main()