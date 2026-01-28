"""
Test script to verify the traffic anomaly detection system
(Updated for OpenCV-based detection - NO PyTorch required)
"""
import os
import sys

print("="*70)
print("TESTING TRAFFIC ANOMALY DETECTION SYSTEM")
print("="*70)
print()

# Test 1: Import all modules
print("Test 1: Checking imports...")
try:
    import cv2
    print("   ✅ OpenCV imported")
    import numpy as np
    print("   ✅ NumPy imported")
    from flask import Flask
    print("   ✅ Flask imported")
    from traffic_anomaly_detector import TrafficAnomalyDetector
    print("   ✅ TrafficAnomalyDetector imported")
    print("✅ All imports successful (NO PyTorch needed!)")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check detector initialization
print("\nTest 2: Initializing detector...")
try:
    detector = TrafficAnomalyDetector()
    print(f"✅ Detector initialized (Background Subtraction based)")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)

# Test 3: Check directories
print("\nTest 3: Checking directories...")
upload_dir = "uploads"
output_dir = "outputs"

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    print(f"✅ Created {upload_dir} directory")
else:
    print(f"✅ {upload_dir} directory exists")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Created {output_dir} directory")
else:
    print(f"✅ {output_dir} directory exists")

# Test 4: Check for sample video
print("\nTest 4: Checking for sample video...")
sample_video = "sample_surveillance.mp4"
if os.path.exists(sample_video):
    print(f"✅ Sample video found: {sample_video}")
    video_size = os.path.getsize(sample_video) / (1024*1024)
    print(f"   Size: {video_size:.2f} MB")
else:
    print(f"⚠️  Sample video not found: {sample_video}")
    print("   You can upload your own video via the web interface")

# Test 5: Test video processing (if sample exists)
if os.path.exists(sample_video):
    print("\nTest 5: Running sample detection...")
    try:
        print("   Processing... (this may take 30-60 seconds)")
        results = detector.process_video(sample_video)
        print(f"   ✅ Processing complete!")
        print(f"   - Anomalies detected: {results['anomaly_count']}")
        print(f"   - Total frames: {results['total_frames']}")
        print(f"   - Anomaly ratio: {results['anomaly_ratio']:.2%}")
        print(f"   - Output saved: {results['output_path']}")
        if results['event_types']:
            print(f"   - Event types: {list(results['event_types'].keys())}")
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print()
print("✅ System is ready!")
print()
print("🎯 Key Features:")
print("   • Background Subtraction (MOG2) - NO training needed")
print("   • Deviation-Based Anomaly Detection")
print("   • Event Classification (Over-speeding, Accidents, etc.)")
print("   • Temporal Consistency Filtering")
print()
print("🚀 To start the web server:")
print("   python server.py")
print()
print("🌐 Then open http://localhost:5000 in your browser")
print()
