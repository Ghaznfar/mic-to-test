# 🎤 Whisper Mic Transcription API

A Flask-based microservice that records audio from your microphone and transcribes it into text using [OpenAI Whisper](https://github.com/openai/whisper). The API supports recording, stopping, and retrieving transcripts — all via simple HTTP requests.

---

## 🚀 Features

- Start and stop microphone recording via API
- Transcribe recorded audio using Whisper
- Built-in CORS support for frontend integration
- Uses `sounddevice` for real-time audio capture
- Simple to run and extend

---

## 🛠 Requirements

- Python 3.8–3.11
- A working microphone
- [ffmpeg](https://ffmpeg.org/download.html) (required by Whisper)



---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/whisper-mic-server.git
cd whisper-mic-server

# Install dependencies
pip install -r requirements.txt
sudo apt update
sudo apt install ffmpeg
