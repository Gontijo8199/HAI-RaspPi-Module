# HAI — Human-AI Interaction Platform

Dispositivo embarcado de apoio escolar: capta a voz do aluno, transcreve localmente e consulta um LLM remoto, exibindo a resposta em display físico. Pensado para rodar em Raspberry Pi sem depender de conexão contínua para o STT.

## Hardware

Raspberry Pi 5 
Raspberry Pi OS Lite

## Pipeline

```
voz → Whisper (local) → LLM API (remoto) → display
```

## Estrutura

```
├── stt/      captura de áudio e transcrição
├── api/      cliente da LLM
├── display/  driver do display
├── core/     orquestração do pipeline
└── config/   configurações e segredos
```

## Modos de operação

```bash
python core/main.py          # detecção automática de voz (VAD)
python core/main.py --ptt    # push-to-talk via Enter
```

## Configuração

Edite `config/settings.toml` para ajustar modelo, idioma e parâmetros de VAD.
Adicione sua chave de API em `config/secrets.toml`, nunca versione esse arquivo.

## Status

Em desenvolvimento.