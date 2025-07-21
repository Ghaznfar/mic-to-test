from flask import Flask, jsonify
from flask_cors import CORS
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import threading
import socket

app = Flask(__name__)
CORS(app)

# Load Whisper model once
model = whisper.load_model("base")

# Global state
recording = False
frames = []
fs = 16000
audio_file_path = None
lock = threading.Lock()


@app.route("/start_record", methods=["GET"])
def start_record():
    global recording, frames

    with lock:
        if recording:
            return jsonify({"status": "error", "message": "Already recording"}), 400

        recording = True
        frames = []

    def _record():
        global frames
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
            while recording:
                data, _ = stream.read(1024)
                frames.append(data)

    threading.Thread(target=_record, daemon=True).start()
    return jsonify({"status": "success", "message": "Recording started"})


@app.route("/stop_record", methods=["GET"])
def stop_record():
    global recording, audio_file_path

    with lock:
        if not recording:
            return jsonify({"status": "error", "message": "Not recording"}), 400

        recording = False

    audio_data = np.concatenate(frames, axis=0)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp_file.name, fs, audio_data)
    audio_file_path = tmp_file.name

    return jsonify({"status": "success", "message": "Recording stopped", "file": audio_file_path})


@app.route("/get_transcript", methods=["GET"])
def get_transcript():
    global audio_file_path

    if not audio_file_path:
        return jsonify({"status": "error", "message": "No audio recorded"}), 400

    try:
        result = model.transcribe(audio_file_path)
        return jsonify({"status": "success", "text": result["text"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = 5000
    host_ip = socket.gethostbyname(socket.gethostname())
    print(f"Mic server is running on http://{host_ip}:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
