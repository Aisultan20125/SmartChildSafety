# -*- coding: utf-8 -*-
"""
guard.py (v8)
=============
SmartChildSafety — система охраны жёлтой коробки + Telegram Web App.
Разработчики: Сарсен Айсултан, Дамир, Богдан (Almaty Youth STEM).

ИЗМЕНЕНИЯ В v8 (относительно v7):
  1. АУДИО (голосовые сообщения):
     - OGG-файлы от Telegram — это контейнер OGG/Opus, а не OGG/Vorbis.
       pygame.mixer нативно Opus не декодирует, поэтому раньше звук мог
       просто не воспроизводиться, если pydub/ffmpeg давали сбой молча.
     - AudioSegment.from_ogg() заменён на AudioSegment.from_file(..., format="ogg"),
       что даёт ffmpeg больше шансов правильно определить кодек (Opus).
     - Экспортируемый WAV принудительно приводится к 44100 Гц / 16 бит /
       стерео — под те же параметры, с которыми инициализирован pygame.mixer,
       чтобы не было рассинхрона и "немого" воспроизведения.
     - При старте скрипта проверяется наличие ffmpeg в PATH (shutil.which),
       с явным предупреждением в консоль, если его нет — раньше это было
       главной скрытой причиной "не играет звук".
     - Добавлен запасной путь воспроизведения напрямую из OGG, если
       конвертация не удалась, и подробное логирование каждого шага.

  2. ДЕТЕКЦИЯ ЖЁЛТОЙ КОРОБКИ НА ВЕСЬ КАДР:
     - HSV-маска и YOLO и раньше строились по всему frame (без ROI по
       центру), поэтому "сужение к центру" было не багом в геометрии,
       а следствием порогов, откалиброванных под крупный план:
         * MIN_AREA=3000 отсекал коробку, если она далеко от камеры
           (в углах комнаты объект в пикселях мельче).
         * YELLOW_LO/HI были рассчитаны на хорошее освещение по центру
           кадра — по углам комнаты света обычно меньше.
         * HUM_R (радиус "человек рядом") был фиксирован в пикселях под
           объект в центре кадра, а не масштабировался под всю сцену.
     - Теперь: MIN_AREA снижен и завязан на разрешение кадра, диапазон
       HSV расширен по S/V (терпимее к тени по углам), HUM_R считается
       как доля диагонали кадра, разрешение захвата увеличено, чтобы
       мелкие/дальние объекты не терялись при бинаризации.

Что умеет:
  • Детекция касания коробки (HSV + MediaPipe + YOLO) с таймером 1.5 сек.
  • "Железный захват" — удержание статуса TAKEN! через YOLO HUMAN.
  • Flask MJPEG-стриминг на :5000 — доступен с любого устройства в сети.
  • Красивый Web App интерфейс для мамы (GET /).
  • Telegram-бот: приветственный пуш с кнопкой Web App + polling входящих.
  • Голосовые сообщения мамы → воспроизведение через динамики ноутбука.
  • Текстовые команды мамы → TTS через pyttsx3 (отказоустойчиво).

Управление:
  TAB / SPACE — FULL ↔ STEALTH
  q           — выход (закрывает окно OpenCV)

Настройка секретов:
  Скопируй .env.example в .env и впиши туда свой TELEGRAM_TOKEN,
  TELEGRAM_CHAT_ID и адрес камеры LOCAL_WEB_URL. Файл .env в Git не
  попадает (он в .gitignore) — так токен бота никогда не утечёт.

Зависимости (ставить по очереди):
  pip install -r requirements.txt
  # pydub нужен ffmpeg: https://ffmpeg.org/download.html
  #   Windows: скачать сборку с https://www.gyan.dev/ffmpeg/builds/ и
  #            добавить папку bin в PATH, либо положить ffmpeg.exe рядом
  #            со скриптом.
  #   macOS:   brew install ffmpeg
  #   Linux:   sudo apt install ffmpeg
"""

import cv2
import time
import threading
import numpy as np
import requests
import torch
import mediapipe as mp
from collections import deque
import io
import os
import sys
import shutil
import queue

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не обязателен, если переменные уже в окружении

# =========================================================================
# 1. НАСТРОЙКИ TELEGRAM
# =========================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print(
        "[Config] ВНИМАНИЕ: TELEGRAM_TOKEN или TELEGRAM_CHAT_ID не заданы.\n"
        "         Скопируй .env.example в .env и заполни своими значениями,\n"
        "         либо задай переменные окружения перед запуском."
    )
    sys.exit(1)

_TG              = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
_TG_TEXT         = f"{_TG}/sendMessage"
_TG_PHOTO        = f"{_TG}/sendPhoto"
_TG_FILE_INFO    = f"{_TG}/getFile"
_TG_UPDATES      = f"{_TG}/getUpdates"

TEMP_PHOTO       = "temp_alert.jpg"
TEMP_VOICE_OGG   = "temp_voice.ogg"
TEMP_VOICE_WAV   = "temp_voice.wav"

ALERT_COOLDOWN   = 15.0     # технический предохранитель API
_last_tg_time    = 0.0
_tg_lock         = threading.Lock()

total_alerts     = 0
_cnt_lock        = threading.Lock()


def _tg_post(url, **kwargs) -> dict:
    """Обёртка над requests.post с таймаутом и обработкой ошибок."""
    try:
        r = requests.post(url, timeout=10, **kwargs)
        return r.json()
    except Exception as e:
        print(f"[TG] Ошибка запроса: {e}")
        return {}


def send_photo_alert(message: str, frame_bgr=None):
    """Фоновая отправка фото-улики (или текста, если кадр недоступен)."""
    def _work():
        if frame_bgr is not None:
            cv2.imwrite(TEMP_PHOTO, frame_bgr)
            with open(TEMP_PHOTO, "rb") as f:
                _tg_post(_TG_PHOTO,
                         data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
                         files={"photo": f})
        else:
            _tg_post(_TG_TEXT, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    threading.Thread(target=_work, daemon=True).start()


def try_alert(message: str, frame_bgr=None) -> bool:
    """Шлёт алерт не чаще раза в ALERT_COOLDOWN секунд."""
    global _last_tg_time, total_alerts
    with _tg_lock:
        if time.time() - _last_tg_time < ALERT_COOLDOWN:
            return False
        _last_tg_time = time.time()
    send_photo_alert(message, frame_bgr)
    with _cnt_lock:
        total_alerts += 1
    return True


def cooldown_active() -> bool:
    return time.time() - _last_tg_time < ALERT_COOLDOWN


# =========================================================================
# 2. ПРИВЕТСТВЕННЫЙ ПУШ + TELEGRAM WEB APP КНОПКА
# =========================================================================
LOCAL_WEB_URL = os.environ.get("LOCAL_WEB_URL", "http://127.0.0.1:5000")

_welcome_sent = False


def send_welcome_and_menu():
    global _welcome_sent
    if _welcome_sent:
        return
    _welcome_sent = True

    text = (
        "🛡 *SmartChildSafety* запущена!\n\n"
        "Система охраны жёлтой коробки активна.\n"
        "Вы получите фото-уведомление, если ребёнок возьмёт коробку.\n\n"
        "📱 *Панель мониторинга (Web App):*\n"
        f"`{LOCAL_WEB_URL}`\n\n"
        "🎤 Отправьте голосовое сообщение — я воспроизведу его в комнате!\n"
        "💬 Или напишите текст — я скажу его вслух.\n\n"
        "_Almaty Youth STEM — Сарсен, Дамир, Богдан_"
    )
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [[
                {
                    "text": "📺 Открыть панель мониторинга",
                    "url": LOCAL_WEB_URL
                }
            ]]
        }
    }
    try:
        requests.post(_TG_TEXT, json=payload, timeout=10)
        print("[TG] Приветственный пуш отправлен.")
    except Exception as e:
        print(f"[TG] Не удалось отправить приветствие: {e}")


# =========================================================================
# 3. АУДИО — воспроизведение голоса мамы
# =========================================================================
_pygame_ok = False
_tts_engine = None

# Явные параметры микшера — те же, к которым мы приводим экспортируемый WAV,
# чтобы не было рассинхрона частоты дискретизации (частая причина "не играет").
_MIXER_FREQ     = 44100
_MIXER_SIZE     = -16     # 16-bit signed
_MIXER_CHANNELS = 2
_MIXER_BUFFER   = 512

try:
    import pygame
    pygame.mixer.init(frequency=_MIXER_FREQ, size=_MIXER_SIZE,
                       channels=_MIXER_CHANNELS, buffer=_MIXER_BUFFER)
    _pygame_ok = True
    print("[Audio] pygame.mixer инициализирован "
          f"({_MIXER_FREQ}Hz, {_MIXER_CHANNELS}ch).")
except Exception as e:
    print(f"[Audio] pygame недоступен: {e}. Воспроизведение отключено.")

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)
    print("[Audio] pyttsx3 инициализирован (TTS готов).")
except Exception as e:
    print(f"[Audio] pyttsx3 недоступен: {e}. TTS отключён.")

# pydub для конвертации ogg(opus) -> wav (нужен ffmpeg в PATH)
_pydub_ok   = False
_ffmpeg_ok  = shutil.which("ffmpeg") is not None
try:
    from pydub import AudioSegment
    _pydub_ok = True
    print("[Audio] pydub доступен — конвертация OGG/Opus→WAV включена.")
except Exception:
    print("[Audio] pydub не найден. Установите: pip install pydub")

if _pydub_ok and not _ffmpeg_ok:
    print("[Audio] ВНИМАНИЕ: ffmpeg не найден в PATH! Голосовые сообщения "
          "не будут декодироваться (Telegram voice = OGG/Opus, нужен ffmpeg). "
          "Установите ffmpeg и перезапустите скрипт — см. инструкцию в шапке файла.")

_audio_queue = queue.Queue()


def _audio_worker():
    """Фоновый поток воспроизведения: ('tts', text) или ('file', path)."""
    while True:
        task = _audio_queue.get()
        if task is None:
            break
        kind, payload = task
        try:
            if kind == "file" and _pygame_ok and os.path.exists(payload):
                pygame.mixer.music.load(payload)
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                print(f"[Audio] Воспроизведение завершено: {payload}")
            elif kind == "tts" and _tts_engine is not None:
                _tts_engine.say(payload)
                _tts_engine.runAndWait()
            else:
                print(f"[Audio] Не могу воспроизвести задачу: {kind}, {payload}")
        except Exception as e:
            print(f"[Audio] Ошибка воспроизведения: {e}")
        _audio_queue.task_done()


_audio_thread = threading.Thread(target=_audio_worker, daemon=True)
_audio_thread.start()


def play_text(text: str):
    _audio_queue.put(("tts", text))


def play_file(path: str):
    _audio_queue.put(("file", path))


def download_and_play_voice(file_id: str):
    """
    Скачивает голосовое OGG/Opus из Telegram, конвертирует в WAV
    (44100Hz/16bit/stereo) через ffmpeg+pydub и воспроизводит.
    Если конвертация не удалась — пробует сыграть OGG напрямую (может не
    сработать для Opus без ffmpeg, но не помешает попробовать).
    """
    def _work():
        try:
            info = requests.get(
                _TG_FILE_INFO, params={"file_id": file_id}, timeout=10
            ).json()
            fpath = info.get("result", {}).get("file_path", "")
            if not fpath:
                print("[Audio] Не удалось получить путь к голосовому файлу.")
                play_text("Не удалось получить голосовое сообщение.")
                return

            url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{fpath}"
            resp = requests.get(url, timeout=15)
            with open(TEMP_VOICE_OGG, "wb") as f:
                f.write(resp.content)
            print(f"[Audio] OGG скачан ({len(resp.content)} байт) → {TEMP_VOICE_OGG}")

            if _pydub_ok and _ffmpeg_ok:
                try:
                    # format="ogg" (а не from_ogg) — даём ffmpeg самому
                    # определить, что внутри Opus, а не Vorbis.
                    seg = AudioSegment.from_file(TEMP_VOICE_OGG, format="ogg")
                    seg = (seg
                           .set_frame_rate(_MIXER_FREQ)
                           .set_channels(_MIXER_CHANNELS)
                           .set_sample_width(2))
                    seg.export(TEMP_VOICE_WAV, format="wav")
                    print("[Audio] Конвертация OGG/Opus → WAV успешна.")
                    play_file(TEMP_VOICE_WAV)
                    return
                except Exception as e:
                    print(f"[Audio] pydub/ffmpeg не смогли конвертировать: {e}. "
                          "Пробуем воспроизвести OGG напрямую.")
            elif not _ffmpeg_ok:
                print("[Audio] ffmpeg отсутствует — пропускаем конвертацию, "
                      "пробуем OGG напрямую (может не сработать для Opus).")

            play_file(TEMP_VOICE_OGG)

        except Exception as e:
            print(f"[Audio] Ошибка при скачивании голосового: {e}")
            play_text("Не удалось воспроизвести голосовое сообщение.")

    threading.Thread(target=_work, daemon=True).start()


# =========================================================================
# 4. TELEGRAM BOT POLLING — обработка входящих сообщений
# =========================================================================
_poll_offset = 0


def _bot_polling():
    global _poll_offset
    print("[Bot] Polling запущен.")
    while True:
        try:
            r = requests.get(
                f"{_TG}/getUpdates",
                params={"offset": _poll_offset, "timeout": 20},
                timeout=25,
            )
            data = r.json()
            updates = data.get("result", [])
            for upd in updates:
                _poll_offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                if not msg:
                    continue
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue

                if "voice" in msg:
                    file_id = msg["voice"]["file_id"]
                    print("[Bot] Получено голосовое сообщение → воспроизводим.")
                    download_and_play_voice(file_id)
                    _tg_post(_TG_TEXT, data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": "🔊 Воспроизвожу ваше голосовое сообщение в комнате!"
                    })

                elif "text" in msg:
                    text = msg["text"].strip()
                    if text.startswith("/"):
                        if text == "/status":
                            status = "🟢 SECURE" if not _session_alert_sent else "🔴 TAKEN!"
                            _tg_post(_TG_TEXT, data={
                                "chat_id": TELEGRAM_CHAT_ID,
                                "text": f"Статус коробки: {status}"
                            })
                        elif text == "/start":
                            send_welcome_and_menu()
                    else:
                        print(f"[Bot] TTS: «{text}»")
                        play_text(text)
                        _tg_post(_TG_TEXT, data={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "text": f"🔊 Говорю вслух в комнате:\n«{text}»"
                        })
        except Exception as e:
            print(f"[Bot] Ошибка polling: {e}")
            time.sleep(3)


# =========================================================================
# 5. ОБЩИЙ БУФЕР ПОСЛЕДНЕГО КАДРА (для Flask-стриминга)
# =========================================================================
_frame_lock   = threading.Lock()
_latest_frame = None


def push_frame(canvas: np.ndarray):
    global _latest_frame
    with _frame_lock:
        _latest_frame = canvas.copy()


def pop_frame() -> bytes | None:
    with _frame_lock:
        if _latest_frame is None:
            return None
        ret, buf = cv2.imencode(".jpg", _latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ret else None


# =========================================================================
# 6. FLASK — стриминг + Web App интерфейс
# =========================================================================
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))

_flask_app = None
try:
    from flask import Flask, Response, render_template_string
    _flask_app = Flask(__name__)
    print("[Flask] Модуль загружен.")
except ImportError:
    print("[Flask] Flask не установлен. Веб-интерфейс недоступен.")
    print("         pip install flask")

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartChildSafety — Панель мониторинга</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a1f2e, #0d1117);
    border-bottom: 2px solid #21262d;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  header .logo { font-size: 28px; }
  header h1 { font-size: 1.4rem; color: #58a6ff; letter-spacing: 0.5px; }
  header p { font-size: 0.8rem; color: #8b949e; margin-top: 2px; }
  .container { max-width: 960px; margin: 0 auto; padding: 24px 16px; display: grid; gap: 20px; }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 1rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; border-bottom: 1px solid #21262d; padding-bottom: 10px; }
  .video-wrapper { position: relative; background: #000; border-radius: 8px; overflow: hidden; text-align: center; }
  .video-wrapper img { width: 100%; max-width: 100%; display: block; border-radius: 8px; }
  .video-badge { position: absolute; top: 10px; left: 10px; background: rgba(239,68,68,0.9); color: #fff; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; letter-spacing: 1px; animation: blink 1s step-start infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .status-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
  .status-dot { width: 14px; height: 14px; border-radius: 50%; background: #3fb950; box-shadow: 0 0 8px #3fb950; animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .status-label { font-size: 1.1rem; font-weight: 600; color: #3fb950; letter-spacing: 0.5px; }
  .status-hint { font-size: 0.82rem; color: #8b949e; margin-top: 4px; }
  .alerts-row { display: flex; align-items: baseline; gap: 10px; }
  .alerts-num { font-size: 2rem; font-weight: 700; color: #f78166; line-height: 1; }
  .alerts-sub { font-size: 0.85rem; color: #8b949e; }
  .uptime-row { display: flex; align-items: baseline; gap: 10px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #21262d; }
  .uptime-num { font-size: 1.5rem; font-weight: 700; color: #58a6ff; font-variant-numeric: tabular-nums; letter-spacing: 2px; }
  .uptime-sub { font-size: 0.85rem; color: #8b949e; }
  footer { text-align: center; padding: 18px; color: #484f58; font-size: 0.78rem; border-top: 1px solid #21262d; margin-top: 8px; }
</style>
<script>
  const _startTs = Date.now();
  function _fmtUptime(ms) {
    const s = Math.floor(ms/1000);
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
    return String(h).padStart(2,'0')+':'+String(m).padStart(2,'0')+':'+String(sec).padStart(2,'0');
  }
  setInterval(() => {
    const el = document.getElementById('uptime');
    if (el) el.textContent = _fmtUptime(Date.now() - _startTs);
  }, 1000);

  setInterval(() => {
    fetch('/api/status').then(r => r.json()).then(d => {
      document.getElementById('alerts-count').textContent = d.alerts;
      const dot   = document.querySelector('.status-dot');
      const label = document.querySelector('.status-label');
      if (d.taken) {
        dot.style.background   = '#f78166';
        dot.style.boxShadow    = '0 0 10px #f78166';
        label.style.color      = '#f78166';
        label.textContent      = '🔴 АЛЕРТ — КОРОБКА ВЗЯТА!';
      } else {
        dot.style.background   = '#3fb950';
        dot.style.boxShadow    = '0 0 8px #3fb950';
        label.style.color      = '#3fb950';
        label.textContent      = '🟢 SECURE — Всё под контролем';
      }
    }).catch(() => {});
  }, 3000);
</script>
</head>
<body>
<header>
  <div class="logo">🛡</div>
  <div>
    <h1>SmartChildSafety — Панель мониторинга</h1>
    <p>ИИ-система охраны в реальном времени • Almaty Youth STEM</p>
  </div>
</header>

<div class="container">

  <div class="card">
    <h2>📡 Прямая трансляция</h2>
    <div class="video-wrapper">
      <img src="/video_feed" alt="AI Camera Stream" />
      <div class="video-badge">● LIVE</div>
    </div>
  </div>

  <div class="card">
    <h2>⚙️ Статус системы</h2>
    <div class="status-row">
      <div class="status-dot"></div>
      <div class="status-label">🟢 SECURE — Всё под контролем</div>
    </div>
    <div class="status-hint">
      Охрана активна. Уведомление придёт, если ребёнок коснётся коробки.
    </div>
  </div>

  <div class="card">
    <h2>🔔 Уведомления</h2>
    <div class="alerts-row">
      <div class="alerts-num" id="alerts-count">0</div>
      <div class="alerts-sub">фото-сообщений отправлено за сессию</div>
    </div>
    <div class="uptime-row">
      <div class="uptime-num" id="uptime">00:00:00</div>
      <div class="uptime-sub">система работает</div>
    </div>
  </div>

</div>
<footer>SmartChildSafety v8 &nbsp;|&nbsp; © 2025 Almaty Youth STEM</footer>
</body>
</html>
"""

if _flask_app is not None:
    @_flask_app.route("/")
    def index():
        return render_template_string(_HTML_TEMPLATE)

    @_flask_app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                jpg = pop_frame()
                if jpg is None:
                    time.sleep(0.05)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
                time.sleep(1 / 30)
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @_flask_app.route("/api/status")
    def api_status():
        from flask import jsonify
        return jsonify({
            "alerts": total_alerts,
            "taken": bool(_session_alert_sent),
        })

    def _run_flask():
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        _flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True, use_reloader=False)


# =========================================================================
# 7. GPU
# =========================================================================
CUDA_OK = torch.cuda.is_available()
if CUDA_OK:
    print(f"[GPU] CUDA активна: {torch.cuda.get_device_name(0)}")
else:
    print("[GPU] CPU-режим")


# =========================================================================
# 8. YOLO
# =========================================================================
YOLO_MODEL_PATH  = "yolo26s.pt"
YOLO_SKIP        = 2
YOLO_CONF        = 0.45
YOLO_BOX_C       = (255, 0, 0)
YOLO_TEXT_C      = (255, 255, 255)

yolo_model       = None
try:
    from ultralytics import YOLO as _YOLO
    yolo_model = _YOLO(YOLO_MODEL_PATH)
    if CUDA_OK:
        yolo_model.to(torch.device("cuda:0"))
    print(f"[YOLO] '{YOLO_MODEL_PATH}' загружена.")
except Exception as e:
    print(f"[YOLO] Не удалось загрузить: {e}")

yolo_all   = []
yolo_human = []


def run_yolo(frame):
    if yolo_model is None:
        return []
    try:
        # imgsz увеличен, чтобы объекты в дальних углах кадра (мелкие в
        # пикселях после ресайза) не терялись при downsize внутри YOLO.
        results = yolo_model.predict(
            frame, device=0 if CUDA_OK else "cpu",
            conf=YOLO_CONF, half=CUDA_OK, imgsz=960, verbose=False)
    except Exception as e:
        print(f"[YOLO] {e}")
        return []
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            lbl = r.names.get(int(b.cls[0].item()), "?")
            if lbl == "person":
                lbl = "HUMAN"
            out.append((x1, y1, x2, y2, lbl))
    return out


def human_near(cx, cy, radius=300):
    for x1, y1, x2, y2, lbl in yolo_human:
        if lbl == "HUMAN":
            hcx, hcy = (x1+x2)/2, (y1+y2)/2
            if ((hcx-cx)**2 + (hcy-cy)**2)**0.5 <= radius:
                return True
    return False


# =========================================================================
# 9. ДЕТЕКЦИЯ ЖЁЛТОЙ КОРОБКИ (на весь кадр)
# =========================================================================
FRAME_W, FRAME_H = 1280, 720   # см. также CAP_PROP_* в main()
_FRAME_DIAG = (FRAME_W**2 + FRAME_H**2) ** 0.5

# Диапазон расширен по S/V вниз — углы комнаты обычно темнее и менее
# насыщены светом, чем центр, где раньше был откалиброван детектор.
YELLOW_LO   = np.array([18, 70, 60])
YELLOW_HI   = np.array([40, 255, 255])

# MIN_AREA снижен: коробка на другом конце комнаты занимает в кадре
# в разы меньше пикселей, чем крупным планом в центре.
MIN_AREA    = 900
MORPH_K     = np.ones((5, 5), np.uint8)   # чуть меньше ядро — не "съедает" мелкие дальние объекты
MIN_ASPECT  = 0.1
MIN_EXTENT  = 0.4
MIN_SOLID   = 0.55
MAX_LOST    = 30

_lost_cnt   = 0
_last_box   = None


def _box_shape_ok(cnt, w, h):
    a = cv2.contourArea(cnt)
    if a <= 0 or w <= 0 or h <= 0:
        return False

    aspect = min(w, h) / max(w, h)
    extent = a / float(w * h)
    hull_a = cv2.contourArea(cv2.convexHull(cnt))
    solid  = a / hull_a if hull_a > 0 else 0

    if aspect < MIN_ASPECT:
        return False

    return extent >= MIN_EXTENT or solid >= MIN_SOLID


def detect_box(frame):
    """
    Ищет жёлтую коробку по ВСЕМУ кадру (весь frame целиком, без обрезки
    по центру/ROI) — от левого верхнего до правого нижнего угла.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    mask = cv2.erode(mask,  MORPH_K, iterations=1)
    mask = cv2.dilate(mask, MORPH_K, iterations=2)
    # RETR_EXTERNAL по полному mask всего кадра = сканирование всей сцены,
    # включая края и углы, а не только центральную область.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, ba = None, 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if _box_shape_ok(c, w, h) and a > ba:
            best, ba = (x, y, w, h), a
    return best


def box_memory(frame):
    global _lost_cnt, _last_box
    det = detect_box(frame)
    if det is not None:
        _last_box, _lost_cnt = det, 0
        return _last_box, True
    if _last_box is not None and _lost_cnt < MAX_LOST:
        _lost_cnt += 1
        return _last_box, False
    _last_box, _lost_cnt = None, 0
    return None, False


# =========================================================================
# 10. ТРИГГЕР ДВИЖЕНИЯ КОРОБКИ
# =========================================================================
_box_hist     = deque(maxlen=5)
SPEED_THRESH  = 550.0
HAND_WIN      = 2.0
_last_hand_t  = 0.0


def box_fast(box, now):
    if box is None:
        return False
    x, y, w, h = box
    _box_hist.append((now, x+w/2, y+h/2))
    if len(_box_hist) < 2:
        return False
    t0, x0, y0 = _box_hist[-2]
    t1, x1, y1 = _box_hist[-1]
    dt = max(t1-t0, 1e-3)
    return ((x1-x0)**2+(y1-y0)**2)**0.5/dt > SPEED_THRESH


def occlude_trigger(box, live, fast, now):
    if now - _last_hand_t > HAND_WIN:
        return False
    return fast or (box is not None and not live and _lost_cnt == 1)


# =========================================================================
# 11. «ЖЕЛЕЗНЫЙ ЗАХВАТ»
# =========================================================================
LOCK_F    = 300
# Радиус "человек рядом" теперь — доля диагонали кадра (~26%), а не
# фиксированные пиксели: так он одинаково хорошо работает и когда объект
# у камеры, и когда сцена — вся комната целиком.
HUM_R     = int(_FRAME_DIAG * 0.26)
_lock_cnt = 0
_lock_ctr = None


def iron_grip(touched, box):
    global _lock_cnt, _lock_ctr
    if touched:
        _lock_cnt = LOCK_F
        if box:
            x, y, w, h = box
            _lock_ctr = (x+w/2, y+h/2)
        return True
    if _lock_cnt > 0:
        if _lock_ctr and not human_near(*_lock_ctr, HUM_R):
            _lock_cnt, _lock_ctr = 0, None
            return False
        _lock_cnt -= 1
        return True
    return False


# =========================================================================
# 12. ТАЙМЕР ПОДТВЕРЖДЕНИЯ (1.5 сек) + СЕССИЯ УДЕРЖАНИЯ
# =========================================================================
CONFIRM_F           = 45
_confirm_cnt        = 0
_session_alert_sent = False


def confirm_timer(raw, grip):
    global _confirm_cnt, _session_alert_sent
    active = raw or grip
    if not active:
        _confirm_cnt, _session_alert_sent = 0, False
        return False
    _confirm_cnt += 1
    if _confirm_cnt >= CONFIRM_F and not _session_alert_sent:
        _session_alert_sent = True
        return True
    return False


# =========================================================================
# 13. MEDIAPIPE HANDS + ЛАЗЕРЫ
# =========================================================================
_mp_hands      = mp.solutions.hands
_mp_draw       = mp.solutions.drawing_utils
_mp_styles     = mp.solutions.drawing_styles

hands_detector = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

_TOUCH_LM = [
    _mp_hands.HandLandmark.WRIST,
    _mp_hands.HandLandmark.THUMB_TIP,
    _mp_hands.HandLandmark.INDEX_FINGER_TIP,
    _mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
]
LASER_D = 150.0
L_FAR   = (0, 200, 0)
L_NEAR  = (0, 0, 255)


def _pt_in(px, py, box):
    x, y, w, h = box
    return x <= px <= x+w and y <= py <= y+h


def touch_check(lml, fw, fh, box):
    if not box or not lml:
        return False
    for ls in lml:
        for lid in _TOUCH_LM:
            lm = ls.landmark[lid]
            if _pt_in(int(lm.x*fw), int(lm.y*fh), box):
                return True
    return False


def draw_lasers(canvas, lml, fw, fh, box):
    if not box or not lml:
        return
    x, y, w, h = box
    bcx, bcy = x+w/2, y+h/2
    for ls in lml:
        wrist = ls.landmark[_mp_hands.HandLandmark.WRIST]
        wx, wy = int(wrist.x*fw), int(wrist.y*fh)
        d = ((wx-bcx)**2+(wy-bcy)**2)**0.5
        cv2.line(canvas, (wx, wy), (int(bcx), int(bcy)),
                 L_NEAR if d <= LASER_D else L_FAR, 1, cv2.LINE_AA)


# =========================================================================
# 14. РЕЖИМЫ ОТОБРАЖЕНИЯ
# =========================================================================
MODE_FULL    = 1
MODE_STEALTH = 2
_mode        = MODE_FULL


def toggle():
    global _mode
    _mode = MODE_STEALTH if _mode == MODE_FULL else MODE_FULL


# =========================================================================
# 15. ОСНОВНОЙ ЦИКЛ
# =========================================================================
def main():
    global _last_hand_t, yolo_all, yolo_human

    if _flask_app is not None:
        ft = threading.Thread(target=_run_flask, daemon=True)
        ft.start()
        print(f"[Flask] Сервер запущен → http://0.0.0.0:{FLASK_PORT}")

    bot_t = threading.Thread(target=_bot_polling, daemon=True)
    bot_t.start()

    threading.Thread(target=send_welcome_and_menu, daemon=True).start()

    # -------------------------------------------------------------------
    # ВЫБОР КАМЕРЫ:
    #   0 — Iriun Webcam по Wi-Fi (по умолчанию)
    #   1 — Iriun Webcam по USB-проводу
    # Если камера физически не видит какой-то угол комнаты (например,
    # из-за узкого угла обзора объектива или цифрового зума в приложении
    # Iriun на телефоне) — это ограничение самой камеры/линзы, программно
    # расширить FOV нельзя; проверьте настройки зума в приложении Iriun.
    # -------------------------------------------------------------------
    cam_index = int(os.environ.get("CAMERA_INDEX", "1"))
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть камеру (индекс {cam_index}). Попробуйте другой CAMERA_INDEX.")
        return

    _session_start = time.time()
    fps_buf   = deque(maxlen=30)
    prev_t    = time.time()
    frame_idx = 0

    print("Guard v8 запущена. TAB/SPACE — FULL/STEALTH | 'q' — выход.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра.")
            break

        # Ресайз к FRAME_W x FRAME_H — это НЕ обрезка по центру, а простое
        # масштабирование всего кадра целиком, поэтому вся сцена (все
        # углы комнаты, которые видит объектив) остаётся в кадре.
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        fh, fw = frame.shape[:2]
        now    = time.time()

        # ---- Коробка (по всему кадру) ----
        box, live = box_memory(frame)
        fast = box_fast(box, now)

        # ---- Руки ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = hands_detector.process(rgb)
        lml    = mp_res.multi_hand_landmarks or []
        if lml:
            _last_hand_t = now

        # ---- YOLO (по всему кадру) ----
        if yolo_model is not None and frame_idx % YOLO_SKIP == 0:
            yolo_all   = run_yolo(frame)
            yolo_human = [b for b in yolo_all if b[4] == "HUMAN"]
        frame_idx += 1

        # ---- Логика тревоги ----
        raw_t  = touch_check(lml, fw, fh, box)
        mov_t  = occlude_trigger(box, live, fast, now)
        touched = raw_t or mov_t
        grip    = iron_grip(touched, box)
        fire    = confirm_timer(touched, grip)

        if fire:
            sent = try_alert(
                "⚠️ Ребёнок взял коробку!",
                frame_bgr=frame,
            )
            if sent:
                print("[Telegram] Фото-улика отправлена.")

        # ---- Рисуем ----
        canvas = frame.copy()

        if _mode == MODE_FULL:
            for x1, y1, x2, y2, lbl in yolo_all:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), YOLO_BOX_C, 2)
                cv2.putText(canvas, lbl, (x1, max(y1-6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, YOLO_TEXT_C, 2)

        draw_lasers(canvas, lml, fw, fh, box)

        for ls in lml:
            _mp_draw.draw_landmarks(
                canvas, ls, _mp_hands.HAND_CONNECTIONS,
                _mp_styles.get_default_hand_landmarks_style(),
                _mp_styles.get_default_hand_connections_style(),
            )
            xs = [int(l.x*fw) for l in ls.landmark]
            ys = [int(l.y*fh) for l in ls.landmark]
            cv2.rectangle(canvas,
                          (min(xs)-10, min(ys)-10),
                          (max(xs)+10, max(ys)+10),
                          (0, 255, 0), 2)

        if box is not None:
            x, y, bw, bh = box
            if grip:
                cv2.rectangle(canvas, (x, y), (x+bw, y+bh), (0, 0, 255), 4)
                cv2.putText(canvas, "TAKEN!", (x, y-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if not live:
                    cv2.putText(canvas, "TARGET: HIDDEN (KEEP LOCK)", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)
                if _mode == MODE_FULL and _lock_cnt > 0:
                    cv2.putText(canvas, f"LOCK: {_lock_cnt} fr", (x, y+bh+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 255), 2)
                if not _session_alert_sent and _confirm_cnt < CONFIRM_F:
                    cv2.putText(canvas, f"CONFIRMING: {_confirm_cnt/30:.1f}s / 1.5s",
                                (x, y+bh+46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)
            else:
                cv2.rectangle(canvas, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
                cv2.putText(canvas, "GUARD: YELLOW BOX", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(canvas, "TARGET: SECURE", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        y_ui = 30
        if cooldown_active():
            rem = ALERT_COOLDOWN - (now - _last_tg_time)
            cv2.putText(canvas, f"COOLDOWN {rem:.0f}s", (10, y_ui),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)
            y_ui += 28
        cv2.putText(canvas, f"MODE: {'FULL' if _mode==MODE_FULL else 'STEALTH'} [TAB]",
                    (10, y_ui), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        atxt  = f"ALERTS: {total_alerts}"
        ts, _ = cv2.getTextSize(atxt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(canvas, atxt, (fw-ts[0]-15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

        fps_buf.append(1.0 / max(now-prev_t, 1e-6))
        prev_t = now
        elapsed = int(now - _session_start)
        h_u, m_u, s_u = elapsed//3600, (elapsed%3600)//60, elapsed%60
        uptime_str = f"{h_u:02d}:{m_u:02d}:{s_u:02d}"
        cv2.putText(canvas, f"FPS: {sum(fps_buf)/len(fps_buf):.1f}  UP: {uptime_str}",
                    (10, fh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        push_frame(canvas)

        cv2.imshow("SmartChildSafety v8", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (9, 32):
            toggle()

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    _audio_queue.put(None)


if __name__ == "__main__":
    main()
