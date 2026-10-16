import os
import tempfile
from fastapi import FastAPI, Request, HTTPException
from faster_whisper import WhisperModel
from gigachat import GigaChat

app = FastAPI()

# --- 1. Загрузка модели Whisper ---
# ВАЖНО: для бесплатного Render (512 МБ RAM) используем "tiny".
# Модель "small" или "base" не влезет в память.
print("Загрузка модели Whisper...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Whisper готов.")

# --- 2. Инициализация клиента GigaChat ---
# verify_ssl_certs=False обязателен, иначе будет SSL-ошибка
# из-за российских сертификатов, которых нет в окружении Render.
giga = GigaChat(
    credentials=os.environ.get("GIGACHAT_KEY"),
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
    timeout=60.0
)

@app.get("/health")
def health():
    """Проверка, что сервер жив."""
    return {"status": "healthy", "whisper": "tiny", "llm": "gigachat"}

@app.post("/voice_assistant")
async def voice_assistant(request: Request):
    """
    Полный цикл: аудио с ESP32 -> текст -> ответ GigaChat -> JSON.
    """
    # 1. Читаем сырое аудио из тела запроса
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Пустой аудиопоток")

    tmp_path = None
    try:
        # 2. Сохраняем во временный WAV-файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 3. Распознаём речь через Whisper
        # vad_filter=True отсекает тишину и шум с микрофона ESP32
        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )
        user_text = " ".join(s.text.strip() for s in segments).strip()

        if not user_text:
            return {"user_text": "", "ai_text": "Не удалось распознать речь"}

        # 4. Отправляем распознанный текст в GigaChat
        # Просим отвечать коротко, чтобы текст влез на OLED-экран
        response = giga.chat(
            f"Ты краткий ассистент. Ответь очень коротко, не более 3 предложений, на вопрос: {user_text}"
        )

        # 5. Извлекаем ответ из структуры GigaChat
        # ВНИМАНИЕ: у GigaChat путь к тексту отличается от OpenAI!
        # Здесь не .choices[0].message.content, а .messages[0].content[0].text
        ai_text = response.messages[0].content[0].text.strip()

        # 6. Возвращаем оба текста на ESP32
        return {"user_text": user_text, "ai_text": ai_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
    finally:
        # 7. Удаляем временный файл
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
