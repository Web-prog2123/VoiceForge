import os
import re
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from secrets import compare_digest
from kokoro import KPipeline

app = FastAPI(title="VoiceForge AI Studio")
security = HTTPBasic()

# ЛОГІН ТА ПАРОЛЬ ДЛЯ ВХОДУ НА САЙТ
ADMIN_USER = "Inmosttruf"
ADMIN_PASS = "#agn4Mb8"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    is_user_correct = compare_digest(credentials.username, ADMIN_USER)
    is_pass_correct = compare_digest(credentials.password, ADMIN_PASS)
    if not (is_user_correct and is_pass_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невірний логін або пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Ініціалізація AI-моделі
print("Ініціалізація AI моделі Kokoro...")
pipeline = KPipeline(lang_code='a')

# Розширений список природних голосів
VOICES = {
    "af_bella": "🇺🇸 AF - Bella (Жіночий, м'який)",
    "af_sky": "🇺🇸 AF - Sky (Жіночий, природний)",
    "af_sarah": "🇺🇸 AF - Sarah (Жіночий, класичний)",
    "af_nicole": "🇺🇸 AF - Nicole (Жіночий, виразний)",
    "am_michael": "🇺🇸 AM - Michael (Чоловічий, глибокий)",
    "am_fenrir": "🇺🇸 AM - Fenrir (Чоловічий, насичений)",
    "am_adam": "🇺🇸 AM - Adam (Чоловічий, чіткий)",
    "am_liam": "🇺🇸 AM - Liam (Чоловічий, енергійний)",
    "bf_emma": "🇬🇧 BF - Emma (Британський жіночий)",
    "bm_george": "🇬🇧 BM - George (Британський чоловічий)"
}

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def split_text(text, max_chars=350):
    text = text.strip()
    if not text:
        return []
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: str = Depends(authenticate)):
    return templates.TemplateResponse(request=request, name="index.html", context={"voices": VOICES})

@app.post("/generate")
async def generate(
    text: str = Form(...),
    voice: str = Form("af_bella"),
    speed: float = Form(1.0),
    user: str = Depends(authenticate)
):
    try:
        chunks = split_text(text)
        if not chunks:
            return JSONResponse({"status": "error", "message": "Введіть текст."})

        all_audio = []
        sample_rate = 24000
        pause = np.zeros(int(sample_rate * 0.25))

        for chunk in chunks:
            generator = pipeline(chunk, voice=voice, speed=speed, split_pattern=r'\n+')
            for gs, ps, audio in generator:
                if audio is not None:
                    all_audio.append(audio)
            all_audio.append(pause)

        if not all_audio:
            return JSONResponse({"status": "error", "message": "Помилка створення аудіо."})

        complete_audio = np.concatenate(all_audio)
        out_path = os.path.join("static", "output.wav")
        sf.write(out_path, complete_audio, sample_rate)

        return JSONResponse({"status": "success", "audio_url": "/static/output.wav"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)