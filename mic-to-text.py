from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from pydub import AudioSegment
import tempfile
import os

app = Flask(__name__)
CORS(app)

model = whisper.load_model("base")

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    audio_file = request.files['audio']

    try:
        # Save .webm file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            audio_file.save(temp_input.name)
            temp_input_path = temp_input.name

        # Convert webm to wav using pydub
        audio_segment = AudioSegment.from_file(temp_input_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            audio_segment.export(temp_wav.name, format="wav")
            temp_wav_path = temp_wav.name

        # Transcribe using Whisper
        result = model.transcribe(temp_wav_path)
        return jsonify({"status": "success", "text": result["text"]})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Safe cleanup after all usage is done
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except: pass
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            try: os.remove(temp_wav_path)
            except: pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
