# 🚦 Traffic Anomaly Detection System - OpenCV Edition

## 🎯 Overview

This is a **lightweight, production-ready** anomaly detection system that uses **OpenCV's Background Subtraction** (MOG2) instead of deep learning models. This approach offers:

- ✅ **NO PyTorch dependency** - Lightweight and easy to deploy
- ✅ **NO training required** - Works out of the box
- ✅ **Fast initialization** - Ready in seconds
- ✅ **Low memory usage** - Runs on modest hardware
- ✅ **Real-time capable** - Process videos quickly
- ✅ **Deviation-based detection** - Learns normal behavior automatically

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- opencv-python
- numpy
- scipy
- Flask (for web interface)
- matplotlib (for visualizations)

### 2. Test the System

```bash
python test_new_system.py
```

### 3. Process a Video

```bash
# Simple usage
python detect_anomalies.py --video sample_surveillance.mp4

# With custom parameters
python detect_anomalies.py --video sample_surveillance.mp4 --output results.mp4 --min_area 1000 --area_sigma 2.5

# Export JSON results
python detect_anomalies.py --video sample_surveillance.mp4 --export_json
```

### 4. Use Programmatically

```python
from traffic_anomaly_detector import TrafficAnomalyDetector

# Initialize detector
detector = TrafficAnomalyDetector(
    min_area=800,      # Minimum object size to track
    area_sigma=3,      # Sensitivity for size anomalies
    speed_sigma=3      # Sensitivity for speed anomalies
)

# Process video
results = detector.process_video("input.mp4", "output.mp4")

print(f"Detected {results['anomaly_count']} anomalies")
print(f"Event types: {results['event_types']}")
```

## 🧠 How It Works

### 1. Background Subtraction (MOG2)
- Automatically learns the background model
- Detects moving objects (foreground)
- Robust to shadows and lighting changes

### 2. Object Tracking
- Simple spatial binning for tracking
- Measures speed and area of objects

### 3. Deviation-Based Detection
- Learns **normal behavior** from first 50+ frames
- Detects anomalies when objects deviate from normal:
  - **Area anomalies**: Unusual object sizes
  - **Speed anomalies**: Unusual movement patterns

### 4. Temporal Consistency
- Only flags persistent anomalies (3+ consecutive frames)
- Reduces false positives

### 5. Event Classification
Detected anomalies are classified into:
- 🚗 Over-Speeding
- 🛑 Vehicle Breakdown
- ⚠️ Sudden Braking
- 💥 Accident / Collision
- 🔀 Rash / Zig-Zag Driving
- ❓ Unusual Activity

## 📁 Project Structure

```
anamoly claysys/
├── traffic_anomaly_detector.py  # Main detection engine
├── detect_anomalies.py          # CLI interface
├── app.py                        # Standalone CLI (alternative)
├── test_new_system.py           # Test script
├── inference/
│   └── postprocess.py           # Post-processing utilities
├── requirements.txt             # Dependencies (NO PyTorch!)
└── outputs/                     # Results directory
```

## ⚙️ Configuration

### Detection Parameters

```python
detector = TrafficAnomalyDetector(
    min_area=800,       # Minimum contour area (pixels)
    area_sigma=3,       # Area anomaly threshold (std devs)
    speed_sigma=3       # Speed anomaly threshold (std devs)
)
```

**Tuning Tips:**
- **Increase `min_area`** to ignore small objects
- **Decrease `area_sigma` or `speed_sigma`** for more sensitive detection
- **Increase `area_sigma` or `speed_sigma`** for fewer false positives

### Background Subtraction Parameters

In `traffic_anomaly_detector.py`:

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Number of frames for background learning
    varThreshold=16,    # Threshold for pixel classification
    detectShadows=True  # Detect and mark shadows
)
```

## 📊 Output Format

### Video Output
- Red bounding boxes around anomalies
- Event type labels
- Saves to `traffic_anomaly_output.mp4`

### Results Dictionary
```python
{
    'output_path': 'path/to/output.mp4',
    'anomaly_count': 42,
    'total_frames': 1200,
    'anomaly_frames': [120, 121, 135, ...],
    'event_types': {
        'Over-Speeding': 15,
        'Vehicle Breakdown': 8,
        'Unusual Activity': 19
    },
    'anomaly_ratio': 0.035,
    'video_info': {
        'fps': 30.0,
        'width': 1920,
        'height': 1080,
        'duration': 40.0
    }
}
```

### JSON Export
```json
{
  "video": "sample.mp4",
  "total_frames": 1200,
  "anomaly_count": 42,
  "anomaly_ratio": 0.035,
  "event_types": {
    "Over-Speeding": 15,
    "Vehicle Breakdown": 8
  },
  "anomaly_frames": [120, 121, 135, ...]
}
```

## 🔄 Migration from Deep Learning Model

### What Changed?

**Old System:**
- ✗ Required PyTorch (large dependency)
- ✗ Needed training phase
- ✗ High memory usage
- ✗ Complex setup

**New System:**
- ✅ Only needs OpenCV
- ✅ No training needed
- ✅ Lightweight
- ✅ Simple setup

### Removed Files/Features
- PyTorch model architecture
- Training scripts
- Model checkpoints
- GPU dependencies

### Updated Files
- `traffic_anomaly_detector.py` - Complete rewrite
- `detect_anomalies.py` - Updated CLI
- `inference/postprocess.py` - Removed PyTorch
- `requirements.txt` - Removed PyTorch

## 🎯 Performance

**Typical Performance:**
- **Initialization**: < 1 second
- **Processing Speed**: ~30-60 FPS (depends on resolution)
- **Memory Usage**: ~200-500 MB
- **CPU Usage**: Moderate (optimized for real-time)

## 🐛 Troubleshooting

### Issue: No anomalies detected
**Solution:** Lower `area_sigma` and `speed_sigma` values

### Issue: Too many false positives
**Solution:** Increase `area_sigma` and `speed_sigma` values, or increase `min_area`

### Issue: Video processing is slow
**Solution:** The system automatically downscales for processing. For faster results, use lower resolution input videos.

## 📝 License

This project is provided as-is for anomaly detection in surveillance videos.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Last Updated:** January 2026
**Version:** 2.0.0 (OpenCV-based)
