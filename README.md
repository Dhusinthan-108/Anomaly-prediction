# 🚦 Traffic Anomaly Detection System
## OpenCV-Based Background Subtraction Edition v2.0

[![Status](https://img.shields.io/badge/status-production_ready-brightgreen)]()
[![Version](https://img.shields.io/badge/version-2.0.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

> **🎉 Major Update:** Migrated from PyTorch deep learning to lightweight OpenCV background subtraction!

---

## ✨ Features

- **✅ NO Training Required** - Works out of the box with zero setup.
- **✅ Lightweight & Fast** - ~200MB dependencies, runs at 30-60 FPS on CPU.
- **✅ Real-time Detection** - Processes live video streams instantly.
- **✅ Rich Event Classification** - Detects over 10 different types of anomalies.
- **✅ Web Interface** - Modern dashboard for easy upload, monitoring, and analysis.
- **✅ Advanced CLI** - Full command-line support with JSON export and tuning.
- **✅ Python API** - Modular design for easy integration.

---

## 🎯 What It Detects

The system automatically classifies anomalies into specific categories:

- **🚗 Over-Speeding** - Vehicles exceeding dynamic speed thresholds.
- **🛑 Vehicle Breakdown** - Stationary vehicles in moving lanes.
- **⚠️ Sudden Braking** - Abrupt deceleration patterns.
- **💥 Accident / Collision** - Unusual object merging or size changes.
- **🔀 Rash / Zig-Zag Driving** - Erratic lane changing and movement.
- **⛔ Wrong-Way Driving** - Movement against the flow of traffic.
- **🔙 Reverse Driving** - Vehicles backing up on the road.
- **🛣️ Lane Violation** - Driving outside designated lane boundaries.
- **🚦 Stopped in Traffic** - Abnormal stops in flow.
- **📦 Object on Road** - Debris or unexpected obstacles.
- **❓ Unusual Activity** - Other statistical anomalies.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Interface (Recommended)
Launch the full dashboard to upload videos and visualize results.
```bash
python server.py
# Open http://localhost:5000 in your browser
```

### 3. Run Command Line Interface
Process videos directly from the terminal.
```bash
python detect_anomalies.py --video input.mp4
```

---

## 📸 Screenshots

### Web Interface Dashboard
_Upload video → Watch real-time analysis → View comprehensive stats_

### Annotated Output
_Red bounding boxes mark anomalies with clear text labels describing the event (e.g., "ANOMALY: Over-Speeding")_

---

## 🏗️ How It Works

1. **Background Subtraction**: Uses MOG2 (Mixture of Gaussians) to separate moving objects from the static background.
2. **Object Tracking**: Tracks objects across frames to calculate speed and trajectory.
3. **Behavior Learning**: The system auto-learns "normal" traffic patterns (speed, size, flow) during the first ~50 frames.
4. **Deviation Detection**: Flags objects that statistically deviate (by 3σ+) from the learned normal behavior.
5. **Event Classification**: logic rules classify the specific type of anomaly based on speed profiles, direction, and size changes.

---

## ⚙️ Configuration & Tuning

You can adjust the sensitivity of the detector:

```python
detector = TrafficAnomalyDetector(
    min_area=800,       # Minimum object size (pixels)
    area_sigma=3.0,     # Area deviation threshold (lower = more sensitive)
    speed_sigma=3.0     # Speed deviation threshold (lower = more sensitive)
)
```

**CLI Tuning:**
```bash
python detect_anomalies.py --video input.mp4 --speed_sigma 2.5 --area_sigma 2.5
```

---

## 📁 Project Structure

```
anamoly claysys/
├── traffic_anomaly_detector.py  # 🧠 CORE ENGINE - Main detection logic & class
├── server.py                    # 🌐 WEB SERVER - Flask backend API
├── detect_anomalies.py          # 💻 CLI TOOL - Command line interface
├── index.html                   # 🎨 FRONTEND - Dashboard UI
├── requirements.txt             # 📦 DEPENDENCIES - Package list
├── inference/                   # 🔧 UTILS - Helper scripts
├── uploads/                     # 📂 DATA - Input video storage
├── outputs/                     # 📂 RESULTS - Processed videos & JSON
└── test_new_system.py           # 🧪 TESTS - Validation script
```

---

## 🌐 Web API Documentation

The Flask server provides a REST API for integration:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload a video file |
| `/api/process` | POST | Start processing a specific file |
| `/api/status/<id>` | GET | Get real-time progress & results |
| `/api/video/<id>` | GET | Stream the processed video |
| `/api/download/<id>` | GET | Download final result |

---

## 📊 Performance Metrics

| Metric | Specification |
|--------|---------------|
| **Architecture** | OpenCV MOG2 + Statistical deviation |
| **Speed** | 30-60 FPS (CPU) |
| **Memory** | ~200MB RAM |
| **Latency** | Real-time (<50ms per frame) |
| **Hardware** | Any modern CPU (No GPU req.) |

---

## 🧪 Testing

Verify the system installation and logic:

```bash
# Run the validation suite
python test_new_system.py

# Run on sample video
python detect_anomalies.py --video sample_surveillance.mp4 --export_json
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## 📄 License

MIT License. Free for academic and commercial use.

---

**Built with ❤️ for Safer Roads.**
_Powered by OpenCV & Python_
