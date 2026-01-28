# ⚡ Quick Start Guide

## 🎯 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test the System
```bash
python test_new_system.py
```

### Step 3: Start Processing!

#### Option A: Web Interface (Recommended)
```bash
# Start the web server
python server.py

# Open your browser to:
# http://localhost:5000
```

#### Option B: Command Line
```bash
# Process a video
python detect_anomalies.py --video sample_surveillance.mp4

# With custom output
python detect_anomalies.py --video input.mp4 --output results.mp4

# Export JSON results
python detect_anomalies.py --video input.mp4 --export_json
```

#### Option C: Python Script
```python
from traffic_anomaly_detector import TrafficAnomalyDetector

# Initialize
detector = TrafficAnomalyDetector()

# Process video
results = detector.process_video("input.mp4", "output.mp4")

# View results
print(f"Detected {results['anomaly_count']} anomalies")
print(f"Event types: {results['event_types']}")
```

## 🎨 Web Interface Guide

1. **Upload Video**
   - Drag & drop or click to select
   - Supports: MP4, AVI, MOV, MKV, WEBM
   - Max size: 500MB

2. **Analyze**
   - Click "Analyze Video"
   - Wait for processing (progress bar shows status)
   - Results appear automatically

3. **Review Results**
   - View annotated video with red boxes
   - Check anomaly statistics
   - See event types detected
   - View timeline of anomalies

4. **Download**
   - Click "Download Result"
   - Save annotated video

## ⚙️ Configuration

### Adjust Detection Sensitivity

```python
detector = TrafficAnomalyDetector(
    min_area=800,      # Default: 800
    area_sigma=3,      # Default: 3 (lower = more sensitive)
    speed_sigma=3      # Default: 3 (lower = more sensitive)
)
```

**Common Adjustments:**
- **More sensitive:** Decrease `area_sigma` and `speed_sigma` to 2.0
- **Less sensitive:** Increase to 4.0 or 5.0
- **Ignore small objects:** Increase `min_area` to 1000 or higher

## 📊 Understanding Results

### Anomaly Count
Total number of anomalous events detected

### Anomaly Ratio
Percentage of frames with anomalies (e.g., 3.5% = 35 anomalous frames out of 1000)

### Event Types
Classified anomalies:
- 🚗 Over-Speeding
- 🛑 Vehicle Breakdown  
- ⚠️ Sudden Braking
- 💥 Accident / Collision
- 🔀 Rash / Zig-Zag Driving
- ❓ Unusual Activity

### Timeline
Visual representation of when anomalies occur in the video

## 🔧 Troubleshooting

### No anomalies detected
- Lower sensitivity: `area_sigma=2, speed_sigma=2`
- Check video has moving objects

### Too many false positives
- Increase sensitivity: `area_sigma=4, speed_sigma=4`
- Increase `min_area=1000`

### Slow processing
- System auto-scales resolution
- Use lower resolution input videos for faster processing

## 💡 Tips

1. **Best results:** Fixed camera, good lighting, clear view
2. **First 50 frames:** System learns "normal" behavior
3. **Temporal consistency:** Anomalies must persist 3+ frames
4. **Event classification:** Automatic based on speed/size patterns

## 📚 More Information

- Full documentation: `README_NEW_SYSTEM.md`
- Migration guide: `MIGRATION.md`
- Example usage: `detect_anomalies.py`

---

**Need help?** Check the documentation or run `python test_system.py` to verify your setup.
