"""
🚦 Traffic Anomaly Detection System
Based on Background Subtraction with Deviation-Based Detection
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
from collections import deque


# =========================================
# 🚦 TRAFFIC EVENT LABELS
# =========================================
EVENT_WRONG_WAY = "Wrong-Way Driving"
EVENT_REVERSE_DRIVING = "Reverse Driving"
EVENT_LANE_VIOLATION = "Lane Violation"
EVENT_OVERSPEED = "Over-Speeding"
EVENT_SUDDEN_BRAKE = "Sudden Braking"
EVENT_RASH_DRIVING = "Rash / Zig-Zag Driving"
EVENT_STOPPED_TRAFFIC = "Stopped in Traffic"
EVENT_ACCIDENT = "Accident / Collision"
EVENT_BREAKDOWN = "Vehicle Breakdown"
EVENT_OBJECT = "Object on Road"
EVENT_UNUSUAL = "Unusual Activity"
EVENT_NORMAL = "Normal Traffic"


# =========================================
# 🚨 EVENT CLASSIFICATION LOGIC
# =========================================
def classify_event(area, speed, area_mean, area_std, speed_mean, speed_std):
    """Classify the type of anomaly based on area and speed statistics."""
    
    # Stopped vehicle or breakdown
    if speed < 0.1:
        return EVENT_BREAKDOWN
    
    # High speed detection
    if speed_mean and speed > speed_mean + 3 * speed_std:
        return EVENT_OVERSPEED
    
    # Sudden speed changes
    if speed_mean and abs(speed - speed_mean) > 2.5 * speed_std:
        return EVENT_SUDDEN_BRAKE
    
    # Unusual object size (could be wrong vehicle type, accident, etc.)
    if area_mean and area > area_mean + 3 * area_std:
        return EVENT_ACCIDENT
    
    # Small fast-moving object
    if area_mean and area < area_mean - 2 * area_std and speed > 1.0:
        return EVENT_RASH_DRIVING
    
    return EVENT_UNUSUAL


# =========================================
# 🎥 Traffic Anomaly Detector Class
# =========================================
class TrafficAnomalyDetector:
    """Complete anomaly detection system using background subtraction and deviation-based detection."""
    
    def __init__(self, min_area=800, area_sigma=3, speed_sigma=3):
        """
        Initialize the anomaly detector.
        
        Args:
            min_area: Minimum contour area to consider
            area_sigma: Number of standard deviations for area anomaly detection
            speed_sigma: Number of standard deviations for speed anomaly detection
        """
        self.min_area = min_area
        self.area_sigma = area_sigma
        self.speed_sigma = speed_sigma
        
        # Background subtractor (NO TRAINING REQUIRED)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Buffers to learn normal behavior
        self.area_history = deque(maxlen=200)
        self.speed_history = deque(maxlen=200)
        self.tracks = {}
        self.anomaly_memory = {}
        
        print(f"🔧 Initialized detector with background subtraction (NO deep learning required)")
    
    def detect_anomaly(self, frame_id, fg_mask):
        """
        Detect anomalies based on deviation from learned normal behavior.
        
        Args:
            frame_id: Current frame index
            fg_mask: Foreground mask from background subtractor
            
        Returns:
            anomalies: List of (x, y, w, h, event_type) tuples
        """
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        anomalies = []

        # Normal behavior statistics (after learning phase)
        area_mean = np.mean(self.area_history) if len(self.area_history) > 50 else None
        area_std = np.std(self.area_history) if len(self.area_history) > 50 else None
        speed_mean = np.mean(self.speed_history) if len(self.speed_history) > 50 else None
        speed_std = np.std(self.speed_history) if len(self.speed_history) > 50 else None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2

            # Simple tracking via spatial binning
            obj_id = (cx // 25, cy // 25)

            speed = 0
            if obj_id in self.tracks:
                px, py, pf = self.tracks[obj_id]
                speed = np.sqrt((cx - px)**2 + (cy - py)**2)

            self.tracks[obj_id] = (cx, cy, frame_id)

            # Learn normal patterns
            self.area_history.append(area)
            self.speed_history.append(speed)

            is_anomaly = False

            # Check for area anomaly
            if area_mean is not None and area_std is not None:
                if abs(area - area_mean) > self.area_sigma * area_std:
                    is_anomaly = True

            # Check for speed anomaly
            if speed_mean is not None and speed_std is not None:
                if abs(speed - speed_mean) > self.speed_sigma * speed_std:
                    is_anomaly = True

            if is_anomaly:
                # Classify the event type
                event_type = classify_event(area, speed, area_mean, area_std, speed_mean, speed_std)
                anomalies.append((x, y, w, h, event_type))

        return anomalies

    def process_video(self, video_path, output_path=None, progress_callback=None):
        """
        Process video and mark anomalies with red boxes.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            progress_callback: Callback function for progress updates
            
        Returns:
            dict: Processing results including anomaly count, frames, and statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_path is None:
            output_path = str(Path(video_path).parent / "traffic_anomaly_output.mp4")

        # Open video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("Could not read video")

        print(f"📹 Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

        # Create output video writer
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        frame_id = 0
        anomaly_counter = 0
        anomaly_frames = []
        event_types = {}

        # Create output frames directory
        output_frames_dir = Path(output_path).parent / "output_frames"
        output_frames_dir.mkdir(exist_ok=True)

        print("🔍 Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Detect anomalies
            anomalies = self.detect_anomaly(frame_id, fg_mask)

            for (x, y, w, h, event_type) in anomalies:
                key = (x // 30, y // 30)
                self.anomaly_memory[key] = self.anomaly_memory.get(key, 0) + 1

                # Temporal consistency (true anomaly)
                if self.anomaly_memory[key] >= 3:
                    # Draw dark red box around anomaly
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 139), 3)
                    cv2.putText(frame, f"ANOMALY: {event_type}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 139), 2)
                    
                    anomaly_counter += 1
                    anomaly_frames.append(frame_id)
                    event_types[event_type] = event_types.get(event_type, 0) + 1

            # Write frame to output
            out.write(frame)

            # Save sample frames
            if frame_id % 30 == 0:
                frame_path = output_frames_dir / f"frame_{frame_id:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)

            frame_id += 1
            
            # Progress callback
            if progress_callback and frame_id % 10 == 0:
                progress = frame_id / total_frames
                progress_callback(f"Processing frame {frame_id}/{total_frames}", progress)
            
            time.sleep(0.001)

        cap.release()
        out.release()

        print(f"✅ Processing completed!")
        print(f"   True anomalies detected: {anomaly_counter}")
        print(f"   Output saved to: {output_path}")
        
        # Return comprehensive results
        return {
            'output_path': output_path,
            'anomaly_count': anomaly_counter,
            'total_frames': total_frames,
            'anomaly_frames': anomaly_frames,
            'event_types': event_types,
            'anomaly_ratio': anomaly_counter / total_frames if total_frames else 0,
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'duration': total_frames / fps if fps else 0
            }
        }


if __name__ == "__main__":
    # Test the detector
    detector = TrafficAnomalyDetector()
    
    if os.path.exists("sample_surveillance.mp4"):
        print("Testing with sample_surveillance.mp4...")
        results = detector.process_video("sample_surveillance.mp4")
        print(f"\n📊 Results:")
        print(f"   Anomaly Count: {results['anomaly_count']}")
        print(f"   Total Frames: {results['total_frames']}")
        print(f"   Anomaly Ratio: {results['anomaly_ratio']:.2%}")
        print(f"   Event Types: {results['event_types']}")
    else:
        print("No sample video found. Please provide a video file to test.")
