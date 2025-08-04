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

# --- Whisper Endpoint ---
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
        for path in [temp_input_path, 'temp_wav_path']:
            if path in locals() and os.path.exists(locals()[path]):
                try: os.remove(locals()[path])
                except: pass

# --- TTS Helper Functions ---
def is_valid_ssml(text):
    try:
        if not re.search(r'<speak[^>]*>', text, re.IGNORECASE):
            text = f"<speak>{text}</speak>"
        ElementTree.fromstring(text)
        return True
    except ElementTree.ParseError:
        return False

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

async def generate_audio(text, voice, path, rate=None, pitch=None, volume=None):
    try:
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
        return True
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        if os.path.exists(path):
            try: os.remove(path)
            except: pass
        return False

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

# --- Routes ---

@app.route("/speak", methods=["POST"])
def speak():
    ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if not rate_limit_check(ip):
        return jsonify({"error": "Rate limit exceeded"}), 429

    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        voice = data.get("voice", "en-US-SteffanNeural")
        rate = data.get("rate", "+0%")
        pitch = data.get("pitch", "+0Hz")
        volume = data.get("volume", "+0%")
        robotic = data.get("robotic", False)

        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": "Text too long"}), 400

        valid_voices = [
            "en-US-SteffanNeural", "en-US-RogerMultilingualNeural",
            "en-US-ChristopherNeural", "en-US-JennyNeural"
        ]
        if voice not in valid_voices:
            return jsonify({"error": "Invalid voice", "valid_voices": valid_voices}), 400

        # Apply mappings
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
            text = re.sub(r'[!?]', '.', text).replace(". ", ". ")

        has_ssml = bool(re.search(r'<[^>]+>', text))
        if has_ssml:
            if not is_valid_ssml(text):
                return jsonify({"error": "Invalid SSML"}), 400
            final_text = text
            rate = pitch = volume = None
        else:
            final_text = text

        filename = f"tts_{uuid.uuid4()}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)

        success = asyncio.run(generate_audio(final_text, voice, filepath, rate, pitch, volume))
        if not success:
            return jsonify({"error": "TTS failed"}), 500

        response = send_file(filepath, mimetype="audio/mpeg", as_attachment=False)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Content-Length"] = str(os.path.getsize(filepath))

        def delayed_cleanup():
            time.sleep(30)
            try:
                if os.path.exists(filepath): os.remove(filepath)
            except: pass

        threading.Thread(target=delayed_cleanup, daemon=True).start()
        return response

    except Exception as e:
        logger.error(f"Error in /speak: {e}")
        return jsonify({"error": "Internal error"}), 500

@app.route("/voices", methods=["GET"])
def voices():
    return jsonify({
        "voices": [
            "en-US-SteffanNeural", "en-US-RogerMultilingualNeural",
            "en-US-ChristopherNeural", "en-US-JennyNeural"
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
    logger.error(f"Global error: {e}")
    return jsonify({"error": "Server error"}), 500

@atexit.register
def exit_cleanup():
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except: pass

# --- Background Thread ---
threading.Thread(target=cleanup_thread, daemon=True).start()

# --- Start ---
if __name__ == "__main__":
    logger.info("Merged Flask App Running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
