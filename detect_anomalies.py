"""
Anomaly Detection Script
Detect anomalies in video files from command line
(Updated for OpenCV-based detection - NO PyTorch required)
"""

import argparse
from pathlib import Path
import json

from traffic_anomaly_detector import TrafficAnomalyDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Detect Anomalies in Video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--min_area', type=int, default=800, help='Minimum contour area')
    parser.add_argument('--area_sigma', type=float, default=3.0, help='Area anomaly sigma')
    parser.add_argument('--speed_sigma', type=float, default=3.0, help='Speed anomaly sigma')
    parser.add_argument('--export_json', action='store_true', help='Export results as JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔍 ANOMALY DETECTION (OpenCV Background Subtraction)")
    print("=" * 70)
    print()
    
    # Setup paths
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"anomaly_{video_path.name}"
    
    print(f"📹 Input video: {video_path}")
    print(f"📁 Output video: {output_path}")
    print(f"⚙️  Min area: {args.min_area}")
    print(f"⚙️  Area sigma: {args.area_sigma}")
    print(f"⚙️  Speed sigma: {args.speed_sigma}")
    print()
    
    # Initialize detector
    print("🔄 Initializing detector...")
    detector = TrafficAnomalyDetector(
        min_area=args.min_area,
        area_sigma=args.area_sigma,
        speed_sigma=args.speed_sigma
    )
    print()
    
    # Detect anomalies
    print("🚀 Processing video...")
    results = detector.process_video(str(video_path), str(output_path))
    print()
    
    # Print results
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Total frames: {results['total_frames']}")
    print(f"Anomalies detected: {results['anomaly_count']}")
    print(f"Anomaly ratio: {results['anomaly_ratio']:.2%}")
    print(f"Video duration: {results['video_info']['duration']:.2f} seconds")
    print()
    
    if results['event_types']:
        print("🎯 Detected Event Types:")
        for event_type, count in sorted(results['event_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"   • {event_type}: {count} occurrences")
        print()
    
    if results['anomaly_frames']:
        print(f"📍 Anomaly Frames (first 20):")
        for i, frame_id in enumerate(results['anomaly_frames'][:20]):
            timestamp = frame_id / results['video_info']['fps']
            print(f"   {i+1}. Frame {frame_id} (t={timestamp:.2f}s)")
        if len(results['anomaly_frames']) > 20:
            print(f"   ... and {len(results['anomaly_frames'])-20} more")
        print()
    
    # Export JSON
    if args.export_json:
        print("💾 Exporting results...")
        export_data = {
            'video': str(video_path),
            'output': str(output_path),
            'total_frames': results['total_frames'],
            'anomaly_count': results['anomaly_count'],
            'anomaly_ratio': results['anomaly_ratio'],
            'event_types': results['event_types'],
            'anomaly_frames': results['anomaly_frames'],
            'video_info': results['video_info']
        }
        
        json_path = output_path.parent / f"results_{video_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported: {json_path}")
        print()
    
    print("=" * 70)
    print("✅ DETECTION COMPLETE!")
    print("=" * 70)
    print(f"📹 Output video: {output_path}")
    print()

if __name__ == "__main__":
    main()
