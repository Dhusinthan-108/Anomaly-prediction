"""
Flask web server for Traffic Anomaly Detection System
(Updated for OpenCV-based detection - NO PyTorch required)
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import threading
from pathlib import Path
import time

from traffic_anomaly_detector import TrafficAnomalyDetector

# =========================================
# Configuration
# =========================================
app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store processing status
processing_status = {}

# =========================================
# Helper Functions
# =========================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def progress_callback(file_id):
    """Returns a closure that updates progress for the given file_id"""
    def callback(message, progress):
        processing_status[file_id] = {
            'status': 'processing',
            'message': message,
            'progress': progress
        }
    return callback

def process_video_async(file_id, input_path, output_path):
    """Process video in background thread"""
    try:
        processing_status[file_id] = {
            'status': 'processing',
            'message': 'Initializing detector...',
            'progress': 0.0
        }
        
        # Initialize detector
        detector = TrafficAnomalyDetector()
        
        # Process video
        results = detector.process_video(
            input_path,
            output_path,
            progress_callback=progress_callback(file_id)
        )
        
        # Mark as completed
        processing_status[file_id] = {
            'status': 'completed',
            'message': 'Processing complete!',
            'progress': 1.0,
            'results': results
        }
        
    except Exception as e:
        processing_status[file_id] = {
            'status': 'error',
            'message': str(e),
            'progress': 0.0
        }

# =========================================
# Routes
# =========================================
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', path)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{ext}")
    file.save(input_path)
    
    # Initialize status
    processing_status[file_id] = {
        'status': 'uploaded',
        'message': 'Video uploaded successfully',
        'progress': 0.0
    }
    
    return jsonify({
        'file_id': file_id,
        'filename': filename,
        'message': 'Upload successful'
    })

@app.route('/api/process', methods=['POST'])
def process_video():
    """Start video processing"""
    data = request.json
    file_id = data.get('file_id')
    
    if not file_id:
        return jsonify({'error': 'No file ID provided'}), 400
    
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    # Find input file
    input_file = None
    for ext in ALLOWED_EXTENSIONS:
        potential_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{ext}")
        if os.path.exists(potential_path):
            input_file = potential_path
            break
    
    if not input_file:
        return jsonify({'error': 'Input file not found'}), 404
    
    # Set output path
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_output.mp4")
    
    # Start processing in background
    thread = threading.Thread(target=process_video_async, args=(file_id, input_file, output_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Processing started',
        'file_id': file_id
    })

@app.route('/api/status/<file_id>', methods=['GET'])
def get_status(file_id):
    """Get processing status"""
    if file_id not in processing_status:
        return jsonify({'error': 'File not found'}), 404
    
    return jsonify(processing_status[file_id])

@app.route('/api/video/<file_id>', methods=['GET'])
def get_video(file_id):
    """Get processed video"""
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_output.mp4")
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'Video not found'}), 404
    
    return send_file(output_path, mimetype='video/mp4')

@app.route('/api/download/<file_id>', methods=['GET'])
def download_video(file_id):
    """Download processed video"""
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_output.mp4")
    
    if not os.path.exists(output_path):
        return jsonify({'error': 'Video not found'}), 404
    
    return send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f'anomaly_detection_{file_id}.mp4'
    )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'OpenCV-based Traffic Anomaly Detection System',
        'version': '2.0.0'
    })

# =========================================
# Main
# =========================================
if __name__ == '__main__':
    print("=" * 70)
    print("🚦 Traffic Anomaly Detection System - Web Server")
    print("=" * 70)
    print()
    print("✅ OpenCV-based detection (NO PyTorch required)")
    print("📁 Upload folder:", UPLOAD_FOLDER)
    print("📁 Output folder:", OUTPUT_FOLDER)
    print()
    print("🌐 Starting server on http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
