from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import whisper
from pydub import AudioSegment
import tempfile
import os
import uuid
import re
import logging
import threading
import time
import asyncio
import edge_tts
import atexit
from datetime import datetime, timedelta
from xml.etree import ElementTree

# --- App Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Whisper Model ---
whisper_model = whisper.load_model("base")

# --- TTS Config ---
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)
MAX_TEXT_LENGTH = 5000
CLEANUP_INTERVAL = 300
FILE_RETENTION_TIME = 600
request_tracker = {}
request_lock = threading.Lock()

# ------------ SSML helpers (simulate <break/>) ------------

_BREAK_RE = re.compile(r"<\s*break\b([^>]*)/?>", re.IGNORECASE)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^\s">]+)')

STRENGTH_TO_MS = {
    "none": 0,
    "x-weak": 100,
    "weak": 200,
    "medium": 500,
    "strong": 800,
    "x-strong": 1200,
}

def _parse_attrs(attr_text: str) -> dict:
    out = {}
    for k, v in _ATTR_RE.findall(attr_text or ""):
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k.lower()] = v
    return out

def _to_ms(val: str) -> int:
    val = val.strip().lower()
    if val.endswith("ms"):
        return max(0, int(float(val[:-2])))
    if val.endswith("s"):
        return max(0, int(float(val[:-1]) * 1000))
    # plain number means ms
    if val.isdigit():
        return int(val)
    return 0

def parse_text_with_breaks(text: str):
    """
    Returns a list of (chunk_text, pause_ms_after_chunk).
    Supports <break time="500ms"/> and <break strength="medium"/>.
    Other tags are stripped.
    """
    # Normalize newlines
    s = text.replace("\r\n", "\n")

    parts = []
    last_idx = 0
    for m in _BREAK_RE.finditer(s):
        # text before this break
        before = s[last_idx:m.start()]
        # strip any other xml-ish tags from 'before'
        before_clean = re.sub(r"</?[^>]+>", "", before)
        if before_clean.strip():
            parts.append((before_clean, 0))  # pause set below

        attrs = _parse_attrs(m.group(1))
        pause_ms = 0
        if "time" in attrs:
            pause_ms = _to_ms(attrs["time"])
        elif "strength" in attrs:
            pause_ms = STRENGTH_TO_MS.get(attrs["strength"].lower(), 0)

        # attach the pause to the previous chunk if any, else insert pure silence chunk
        if parts:
            chunk_text, _ = parts[-1]
            parts[-1] = (chunk_text, pause_ms)
        else:
            # start with silence if the break is first
            parts.append(("", pause_ms))

        last_idx = m.end()

    # trailing text
    tail = s[last_idx:]
    tail_clean = re.sub(r"</?[^>]+>", "", tail)
    if tail_clean.strip():
        parts.append((tail_clean, 0))

    # If nothing left (e.g., only breaks), keep at least one silence so the output isn't empty
    if not parts:
        parts = [("", 0)]

    return parts

# ------------ Rate limiting / housekeeping ------------

def rate_limit_check(ip):
    current_time = datetime.now()
    with request_lock:
        request_tracker[ip] = [
            t for t in request_tracker.get(ip, [])
            if current_time - t < timedelta(hours=1)
        ]
        if len(request_tracker[ip]) >= 60:
            return False
        request_tracker[ip].append(current_time)
        return True

async def tts_to_file(text, voice, path, rate=None, pitch=None, volume=None):
    """
    Synthesize ONE chunk (plain text only) to 'path' with edge-tts.
    """
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate or "+0%",
        pitch=pitch or "+0Hz",
        volume=volume or "+0%"
    )
    with open(path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

def synthesize_with_pauses(full_text, voice, outfile_path, rate, pitch, volume):
    """
    Parse <break/> tags, synthesize text chunks, and stitch with silence.
    """
    segments = parse_text_with_breaks(full_text)

    # Synthesize each chunk to a temp mp3, then concatenate with silence
    combined = AudioSegment.silent(duration=0)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    temp_paths = []
    try:
        for i, (chunk_text, pause_ms) in enumerate(segments):
            if chunk_text.strip():
                tmp_mp3 = os.path.join(TEMP_DIR, f"seg_{uuid.uuid4()}.mp3")
                temp_paths.append(tmp_mp3)
                loop.run_until_complete(tts_to_file(chunk_text, voice, tmp_mp3, rate, pitch, volume))
                audio = AudioSegment.from_file(tmp_mp3)
                combined += audio
            if pause_ms > 0:
                combined += AudioSegment.silent(duration=pause_ms)

        combined.export(outfile_path, format="mp3")
    finally:
        loop.close()
        # cleanup temp segments
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

# ------------ Whisper Endpoint ------------

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            audio_file.save(temp_input.name)
            temp_input_path = temp_input.name

        audio_segment = AudioSegment.from_file(temp_input_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            temp_wav_path = temp_wav.name

        result = whisper_model.transcribe(temp_wav_path)
        return jsonify({"status": "success", "text": result["text"]})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        for path in ["temp_input_path", "temp_wav_path"]:
            if path in locals():
                try:
                    p = locals()[path]
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

# ------------ TTS Route ------------

@app.route("/speak", methods=["POST"])
def speak():
    ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if not rate_limit_check(ip):
        return jsonify({"error": "Rate limit exceeded"}), 429

    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        voice = data.get("voice", "en-US-SteffanNeural")
        rate = data.get("rate", "+0%")
        pitch = data.get("pitch", "+0Hz")
        volume = data.get("volume", "+0%")
        robotic = data.get("robotic", False)

        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": "Text too long"}), 400

        # Validate voice (extend this list or fetch dynamically if you like)
        valid_voices = [
            "en-US-SteffanNeural",
            "en-US-RogerMultilingualNeural",
            "en-US-ChristopherNeural",
            "en-US-JennyNeural",
            "en-ZA-LeahNeural",
        ]
        if voice not in valid_voices:
            return jsonify({"error": "Invalid voice", "valid_voices": valid_voices}), 400

        # Map friendly controls to edge-tts params
        rate_map = {"x-slow": "-50%", "slow": "-25%", "medium": "+0%", "fast": "+25%", "x-fast": "+50%"}
        pitch_map = {"x-low": "-50Hz", "low": "-25Hz", "medium": "+0Hz", "high": "+25Hz", "x-high": "+50Hz"}
        volume_map = {"silent": "-100%", "x-soft": "-50%", "soft": "-25%", "medium": "+0%", "loud": "+25%", "x-loud": "+50%"}
        rate = rate_map.get(rate, rate)
        pitch = pitch_map.get(pitch, pitch)
        volume = volume_map.get(volume, volume)

        if robotic:
            rate = "+20%"
            pitch = "-20Hz"
            volume = "+10%"
            text = re.sub(r'[!?]', '.', text).replace(".  ", ". ")

        # If text looks like SSML, just use it as hints: we only support <break/> explicitly.
        # Strip outer <speak> if present to avoid leaving empty tags in chunks
        if re.search(r"<\s*speak\b", text, re.IGNORECASE):
            text = re.sub(r"</?\s*speak\b[^>]*>", "", text, flags=re.IGNORECASE)

        filename = f"tts_{uuid.uuid4()}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)

        # Synthesize with simulated <break/> pauses
        synthesize_with_pauses(text, voice, filepath, rate, pitch, volume)

        response = send_file(filepath, mimetype="audio/mpeg", as_attachment=False)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Content-Length"] = str(os.path.getsize(filepath))

        def delayed_cleanup():
            time.sleep(30)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

        threading.Thread(target=delayed_cleanup, daemon=True).start()
        return response

    except Exception as e:
        logger.exception("Error in /speak")
        return jsonify({"error": "Internal error"}), 500

# ------------ Voices / Health ------------

@app.route("/voices", methods=["GET"])
def voices():
    # You can also fetch dynamically via edge_tts.list_voices(), but that requires async & caching.
    return jsonify({
        "voices": [
            "en-US-SteffanNeural",
            "en-US-RogerMultilingualNeural",
            "en-US-ChristopherNeural",
            "en-US-JennyNeural",
            "en-ZA-LeahNeural",
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "temp_files": len(os.listdir(TEMP_DIR))
    })

@app.errorhandler(Exception)
def global_error(e):
    logger.exception("Global error")
    return jsonify({"error": "Server error"}), 500

@atexit.register
def exit_cleanup():
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except Exception:
            pass

def cleanup_old_files():
    now = time.time()
    for f in os.listdir(TEMP_DIR):
        fp = os.path.join(TEMP_DIR, f)
        if os.path.isfile(fp) and now - os.path.getctime(fp) > FILE_RETENTION_TIME:
            try:
                os.remove(fp)
                logger.info(f"Deleted old file: {fp}")
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

def cleanup_thread():
    while True:
        time.sleep(CLEANUP_INTERVAL)
        cleanup_old_files()

# --- Background Thread ---
threading.Thread(target=cleanup_thread, daemon=True).start()

# --- Start ---
if __name__ == "__main__":
    logger.info("Merged Flask App Running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
