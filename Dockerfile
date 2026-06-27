# Build para Raspberry Pi 5 (arm64)
FROM python:3.12-slim

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# secrets.toml deve ser montado em runtime
VOLUME ["/app/config"]

CMD ["python", "core/main.py"]