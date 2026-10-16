FROM python:3.11-slim

WORKDIR /app

# ffmpeg нужен faster-whisper для обработки аудио
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Render сам подставит нужный порт через переменную $PORT
CMD uvicorn server:app --host 0.0.0.0 --port $PORT