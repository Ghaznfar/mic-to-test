from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import edge_tts
import asyncio
import os
import uuid
import re
import logging
import tempfile
import atexit
import threading
import time
from xml.etree import ElementTree
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web app integration

# Configuration
TEMP_DIR = "temp_audio"
MAX_TEXT_LENGTH = 5000  # Prevent abuse
CLEANUP_INTERVAL = 300  # Clean up files every 5 minutes
FILE_RETENTION_TIME = 600  # Keep files for 10 minutes
MAX_CONCURRENT_REQUESTS = 10

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Request tracking for rate limiting
request_tracker = {}
request_lock = threading.Lock()

def cleanup_old_files():
    """Clean up old audio files to prevent disk space issues"""
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getctime(filepath)
                if file_age > FILE_RETENTION_TIME:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def cleanup_thread():
    """Background thread for periodic cleanup"""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        cleanup_old_files()

def is_valid_ssml(text):
    """Validate SSML structure using proper XML parsing"""
    try:
        if not re.search(r'<speak[^>]*>', text, re.IGNORECASE):
            text = f"<speak>{text}</speak>"
        ElementTree.fromstring(text)
        return True
    except ElementTree.ParseError:
        return False

def rate_limit_check(client_ip):
    """Simple rate limiting - 60 requests per hour per IP"""
    current_time = datetime.now()
    with request_lock:
        if client_ip not in request_tracker:
            request_tracker[client_ip] = []
        
        # Remove requests older than 1 hour
        request_tracker[client_ip] = [
            req_time for req_time in request_tracker[client_ip]
            if current_time - req_time < timedelta(hours=1)
        ]
        
        # Check if limit exceeded
        if len(request_tracker[client_ip]) >= 60:
            return False
        
        # Add current request
        request_tracker[client_ip].append(current_time)
        return True

@app.errorhandler(Exception)
def handle_error(e):
    """Global error handler"""
    logger.error(f"Unhandled error: {e}")
    return jsonify({"error": "Internal server error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "temp_files": len(os.listdir(TEMP_DIR))
    })

@app.route("/voices", methods=["GET"])
def get_voices():
    """Get available voices"""
    valid_voices = [
        "en-US-SteffanNeural",
        "en-US-RogerMultilingualNeural", 
        "en-US-ChristopherNeural",
        "en-US-JennyNeural"
    ]
    return jsonify({"voices": valid_voices})

@app.route("/speak", methods=["POST"])
def speak():
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    
    # Rate limiting
    if not rate_limit_check(client_ip):
        return jsonify({"error": "Rate limit exceeded. Maximum 60 requests per hour."}), 429
    
    try:
        # Validate content type
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        
        # Input validation
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"Text too long. Maximum {MAX_TEXT_LENGTH} characters allowed."}), 400

        voice = data.get("voice", "en-US-SteffanNeural")
        rate = data.get("rate", "+0%")
        pitch = data.get("pitch", "+0Hz")
        volume = data.get("volume", "+0%")
        robotic = data.get("robotic", False)

        valid_voices = ["en-US-SteffanNeural", "en-US-RogerMultilingualNeural",
                        "en-US-ChristopherNeural", "en-US-JennyNeural"]
        if voice not in valid_voices:
            return jsonify({
                "error": f"Voice '{voice}' not allowed",
                "valid_voices": valid_voices
            }), 400

        # Convert rate/pitch/volume from word values to percentage/Hz values
        rate_map = {"x-slow": "-50%", "slow": "-25%", "medium": "+0%", "fast": "+25%", "x-fast": "+50%"}
        pitch_map = {"x-low": "-50Hz", "low": "-25Hz", "medium": "+0Hz", "high": "+25Hz", "x-high": "+50Hz"}
        volume_map = {"silent": "-100%", "x-soft": "-50%", "soft": "-25%", "medium": "+0%", "loud": "+25%", "x-loud": "+50%"}
        
        # Convert word values to actual values if needed
        if rate in rate_map:
            rate = rate_map[rate]
        if pitch in pitch_map:
            pitch = pitch_map[pitch]
        if volume in volume_map:
            volume = volume_map[volume]

        # Apply robotic voice settings
        if robotic:
            # Robotic characteristics: monotone pitch, faster rate, precise pronunciation
            if rate == "+0%" or rate == "medium":
                rate = "+20%"  # Faster for robotic feel
            if pitch == "+0Hz" or pitch == "medium":
                pitch = "-20Hz"  # Lower, more monotone
            if volume == "+0%" or volume == "medium":
                volume = "+10%"  # Slightly louder for mechanical effect

        has_ssml = re.search(r'<[^>]+>', text) is not None

        if has_ssml:
            if not is_valid_ssml(text):
                return jsonify({
                    "error": "Invalid SSML structure",
                    "details": "Make sure all tags are properly closed and nested",
                    "example": "<speak>Hello <break time='500ms'/>world</speak>"
                }), 400
            # For SSML input, pass it directly but ignore rate/pitch/volume parameters
            final_text = text
            rate = pitch = volume = None  # Don't use prosody parameters with custom SSML
        else:
            # For plain text, add robotic formatting if requested
            if robotic:
                # Add periods and spaces to make speech more robotic/mechanical
                # Replace natural speech patterns with more mechanical ones
                processed_text = text.replace("!", ".").replace("?", ".")
                # Add slight pauses between sentences for robotic effect
                processed_text = processed_text.replace(". ", ". ")
                final_text = processed_text
            else:
                final_text = text

        # Generate unique filename
        filename = f"tts_{uuid.uuid4()}.mp3"
        filepath = os.path.join(TEMP_DIR, filename)

        # Generate audio
        success = asyncio.run(generate_audio(final_text, voice, filepath, rate, pitch, volume))
        
        if not success:
            return jsonify({"error": "Failed to generate audio"}), 500

        # Verify file was created and has content
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({"error": "Audio generation failed"}), 500

        # Return the file as blob with proper headers
        response = send_file(
            filepath,
            mimetype="audio/mpeg",
            as_attachment=False,  # Don't force download, allow blob handling
            download_name=f"speech_{int(time.time())}.mp3"
        )
        
        # Add headers for better web app integration
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Content-Length'] = str(os.path.getsize(filepath))
        
        # Schedule file cleanup after response is sent
        def cleanup_file():
            time.sleep(30)  # Wait 30 seconds before cleanup
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Cleaned up file: {filename}")
            except Exception as e:
                logger.error(f"Error cleaning up file {filename}: {e}")
        
        threading.Thread(target=cleanup_file, daemon=True).start()
        
        return response

    except Exception as e:
        logger.error(f"Error in speak endpoint: {e}")
        return jsonify({"error": "Failed to process request"}), 500

async def generate_audio(text, voice, path, rate=None, pitch=None, volume=None):
    """Generate audio file using Edge TTS"""
    try:
        # Create Communicate object with prosody parameters
        if rate or pitch or volume:
            communicate = edge_tts.Communicate(
                text=text, 
                voice=voice,
                rate=rate if rate else "+0%",
                pitch=pitch if pitch else "+0Hz", 
                volume=volume if volume else "+0%"
            )
        else:
            # For SSML or when no prosody changes needed
            communicate = edge_tts.Communicate(text=text, voice=voice)
            
        with open(path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                    
        return True
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        # Clean up partial file if it exists
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
        return False

# Start cleanup thread
cleanup_thread_instance = threading.Thread(target=cleanup_thread, daemon=True)
cleanup_thread_instance.start()

# Cleanup on exit
@atexit.register
def cleanup_on_exit():
    """Clean up all temp files on application exit"""
    try:
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        logger.info("Cleaned up all temp files on exit")
    except Exception as e:
        logger.error(f"Error during exit cleanup: {e}")

if __name__ == "__main__":
    logger.info("🚀 Edge TTS Production API starting...")
    logger.info("✅ Features enabled:")
    logger.info("   - CORS support for web apps")
    logger.info("   - Rate limiting (60 req/hour per IP)")
    logger.info("   - Automatic file cleanup")
    logger.info("   - Health check endpoint")
    logger.info("   - Blob response support")
    logger.info("   - Error handling & logging")
    logger.info("🌐 API running on http://0.0.0.0:5001")
    
    app.run(host="0.0.0.0", port=5001, debug=False)