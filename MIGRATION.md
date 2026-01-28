# 🔄 Migration Guide: Deep Learning → OpenCV Background Subtraction

## Overview

This guide documents the complete migration from a PyTorch-based deep learning model to an OpenCV-based background subtraction approach for traffic anomaly detection.

## What Changed?

### Before (Deep Learning Approach)
- ❌ Required PyTorch + TorchVision (>2GB installation)
- ❌ Needed GPU for good performance
- ❌ Required training phase (15-25 epochs)
- ❌ Complex model architecture (EfficientNet-B0 + ConvLSTM Autoencoder)
- ❌ High memory usage (1-2GB+ RAM)
- ❌ Slow initialization (load model, compile, etc.)

### After (OpenCV Background Subtraction)
- ✅ Only requires OpenCV + NumPy (lightweight)
- ✅ CPU-only, no GPU needed
- ✅ NO training required (ready to use)
- ✅ Simple algorithm (MOG2 background subtractor)
- ✅ Low memory usage (~200-500MB)
- ✅ Fast initialization (< 1 second)

## Files Changed

### 1. `traffic_anomaly_detector.py` (Complete Rewrite)
**Old:** Convolutional Autoencoder with optical flow
```python
class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
```

**New:** Background Subtraction with deviation-based detection
```python
self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
```

### 2. `requirements.txt` (Dependencies)
**Removed:**
- `torch>=2.0.0`
- `torchvision>=0.15.0`

**Added:**
- `scipy>=1.11.0` (for gaussian filtering in post-processing)

**Kept:**
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `Flask>=2.3.0`
- `matplotlib>=3.7.0`

### 3. `inference/postprocess.py` (Updated)
**Removed:** PyTorch tensor handling
```python
# Old
if isinstance(scores, torch.Tensor):
    scores = scores.cpu().numpy()
```

**New:** Pure NumPy
```python
# New
if not isinstance(scores, np.ndarray):
    scores = np.array(scores)
```

### 4. `detect_anomalies.py` (Updated)
**Old:** Used PyTorch model with checkpoints
```python
from inference import AnomalyDetector
detector = AnomalyDetector(model_path=model_path)
```

**New:** Uses OpenCV detector (no checkpoints needed)
```python
from traffic_anomaly_detector import TrafficAnomalyDetector
detector = TrafficAnomalyDetector(min_area=800, area_sigma=3, speed_sigma=3)
```

### 5. New Files Added
- `server.py` - Flask web server for the new system
- `test_new_system.py` - Tests for the new OpenCV-based system
- `README_NEW_SYSTEM.md` - Documentation for the new system
- `MIGRATION.md` - This file

## How to Migrate

### Step 1: Install New Dependencies
```bash
# Uninstall old dependencies (optional)
pip uninstall torch torchvision

# Install new requirements
pip install -r requirements.txt
```

### Step 2: Test the New System
```bash
# Run the test script
python test_new_system.py

# Or use the standalone test
python test_system.py
```

### Step 3: Update Your Code

If you were using the old detector:
```python
# Old code
from inference import AnomalyDetector
detector = AnomalyDetector(model_path='best_model.pth')
results = detector.detect_video('input.mp4')
```

Replace with:
```python
# New code
from traffic_anomaly_detector import TrafficAnomalyDetector
detector = TrafficAnomalyDetector(min_area=800, area_sigma=3, speed_sigma=3)
results = detector.process_video('input.mp4', 'output.mp4')
```

### Step 4: Run the Web Server
```bash
python server.py
```

Then open http://localhost:5000 in your browser.

## API Changes

### Old API (Deep Learning)
```python
detector.detect_video(
    video_path='input.mp4',
    return_details=True
)
```

### New API (OpenCV)
```python
detector.process_video(
    video_path='input.mp4',
    output_path='output.mp4',  # New: specify output
    progress_callback=None      # New: progress updates
)
```

### Results Format

**Similar structure, different event types:**

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
    'video_info': {...}
}
```

## Performance Comparison

| Metric | Old (Deep Learning) | New (OpenCV) |
|--------|-------------------|--------------|
| Installation Size | ~3GB | ~200MB |
| Initialization Time | 5-10 seconds | <1 second |
| Memory Usage | 1-2GB | 200-500MB |
| GPU Required | Recommended | No |
| Training Required | Yes (15-25 epochs) | No |
| Processing Speed | 20-30 FPS | 30-60 FPS |
| Accuracy | High | Moderate-High |

## Detection Algorithm Comparison

### Old: Autoencoder Reconstruction Error
1. Extract optical flow
2. Encode flow to latent space
3. Decode latent space
4. Compare reconstruction error
5. Threshold for anomalies

### New: Background Subtraction + Deviation
1. Learn background model (MOG2)
2. Detect foreground objects
3. Track object properties (area, speed)
4. Learn normal behavior statistics
5. Detect deviations from normal
6. Classify event types

## Advantages of New System

1. **No Training Required** - Works out of the box
2. **Lightweight** - Smaller dependencies, faster setup
3. **Faster** - No deep learning inference overhead
4. **Real-time Capable** - Can process live video streams
5. **Easier to Deploy** - Fewer dependencies, no GPU needed
6. **Transparent** - Easier to understand and debug
7. **Configurable** - Easy parameter tuning

## Disadvantages of New System

1. **Lower Accuracy** - May miss subtle anomalies
2. **Less Sophisticated** - No learned representations
3. **Weather Sensitive** - May struggle with rain, snow, etc.
4. **Static Cameras Only** - Assumes fixed camera position
5. **Calibration Needed** - May need parameter tuning per scene

## When to Use Which?

### Use New System (OpenCV) When:
- ✅ You need fast deployment
- ✅ You don't have labeled training data
- ✅ You want lightweight solution
- ✅ You have fixed camera positions
- ✅ You need real-time processing

### Use Old System (Deep Learning) When:
- ✅ You have labeled training data
- ✅ You need maximum accuracy
- ✅ You have access to GPUs
- ✅ You can afford longer setup time
- ✅ You need to detect subtle anomalies

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:** The new system doesn't need PyTorch. Install new requirements:
```bash
pip install -r requirements.txt
```

### Issue: Import errors from old files
**Solution:** Update your imports to use the new system:
```python
from traffic_anomaly_detector import TrafficAnomalyDetector
```

### Issue: Old checkpoint files not working
**Solution:** The new system doesn't use checkpoints. Remove `model_path` arguments.

### Issue: Different results than before
**Solution:** The new algorithm works differently. Tune parameters:
```python
detector = TrafficAnomalyDetector(
    min_area=800,      # Adjust minimum object size
    area_sigma=3,      # Adjust area sensitivity (lower = more sensitive)
    speed_sigma=3      # Adjust speed sensitivity (lower = more sensitive)
)
```

## Rollback Instructions

If you need to rollback to the old system:

1. Restore old files from git history
2. Reinstall PyTorch:
   ```bash
   pip install torch torchvision
   ```
3. Restore old checkpoint files
4. Use old detector API

## Support

For questions or issues:
1. Check `README_NEW_SYSTEM.md`
2. Review test scripts: `test_new_system.py`, `test_system.py`
3. Check example usage in `detect_anomalies.py`

---

**Migration Date:** January 2026  
**Version:** 2.0.0 (OpenCV-based)
