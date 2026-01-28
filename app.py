import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import argparse

class AnomalyDetector:
    """Complete anomaly detection system using background subtraction and deviation-based detection."""
    
    def __init__(self, min_area=800, area_sigma=3, speed_sigma=3):
        self.min_area = min_area
        self.area_sigma = area_sigma
        self.speed_sigma = speed_sigma
        
        # Background subtractor
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
        
    def detect_anomaly(self, frame_id, fg_mask):
        """Detect anomalies based on deviation from learned normal behavior."""
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

            if area_mean is not None:
                if abs(area - area_mean) > self.area_sigma * area_std:
                    is_anomaly = True

            if speed_mean is not None:
                if abs(speed - speed_mean) > self.speed_sigma * speed_std:
                    is_anomaly = True

            if is_anomaly:
                anomalies.append((x, y, w, h))

        return anomalies

    def process_video(self, video_path, output_path):
        """Process video and mark anomalies with red boxes."""
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        frame_id = 0
        anomaly_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fg_mask = self.bg_subtractor.apply(frame)
            anomalies = self.detect_anomaly(frame_id, fg_mask)

            for (x, y, w, h) in anomalies:
                key = (x // 30, y // 30)
                self.anomaly_memory[key] = self.anomaly_memory.get(key, 0) + 1

                # Temporal consistency (true anomaly)
                if self.anomaly_memory[key] >= 3:
                    # Draw red box around anomaly
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "ANOMALY",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    anomaly_counter += 1

            out.write(frame)

            frame_id += 1

        cap.release()
        out.release()

        return anomaly_counter

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection in Videos')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output_anomaly.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Initialize detector
    detector = AnomalyDetector()
    
    print(f"Processing video: {args.input}")
    anomaly_count = detector.process_video(args.input, args.output)
    print(f"Processing completed. Anomalies detected: {anomaly_count}")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    # If no command line args, show usage
    import sys
    if len(sys.argv) == 1:
        print("Usage: python app.py --input <input_video_path> --output <output_video_path>")
        print("\nExample: python app.py --input sample_video.mp4 --output output_anomaly.mp4")
    else:
        main()