"""
Anomaly Detection Script
Detect anomalies in video files from command line
"""

import argparse
from pathlib import Path
import json

from inference import AnomalyDetector, VideoAnnotator
from config import CHECKPOINT_DIR, OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Detect Anomalies in Video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
    parser.add_argument('--annotate', action='store_true', help='Create annotated video')
    parser.add_argument('--export_json', action='store_true', help='Export results as JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔍 ANOMALY DETECTION")
    print("=" * 70)
    print()
    
    # Setup paths
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if args.model:
        model_path = args.model
    else:
        model_path = CHECKPOINT_DIR / 'best_model.pth'
        if not model_path.exists():
            print("⚠️ No trained model found. Using untrained model.")
            model_path = None
    
    print(f"📹 Input video: {video_path}")
    print(f"📁 Output directory: {output_dir}")
    if model_path:
        print(f"🧠 Model: {model_path}")
    print(f"🎯 Threshold: {args.threshold}")
    print()
    
    # Initialize detector
    print("🔄 Initializing detector...")
    detector = AnomalyDetector(model_path=model_path)
    detector.config['threshold'] = args.threshold
    print()
    
    # Detect anomalies
    print("🚀 Detecting anomalies...")
    results = detector.detect_video(str(video_path), return_details=True)
    print()
    
    # Print results
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Total frames: {results['num_frames']}")
    print(f"Anomalies detected: {results['num_anomalies']}")
    print(f"Anomaly ratio: {results['anomaly_ratio']:.2%}")
    print(f"Threshold: {results['threshold']:.4f}")
    print(f"Max score: {results['max_score']:.4f}")
    print(f"Mean score: {results['mean_score']:.4f}")
    print(f"Anomaly segments: {len(results['segments'])}")
    print()
    
    if results['segments']:
        print("🎯 Anomaly Segments:")
        for i, (start, end) in enumerate(results['segments'][:10]):  # Show first 10
            print(f"   {i+1}. Frames {start}-{end} ({end-start} frames)")
        if len(results['segments']) > 10:
            print(f"   ... and {len(results['segments'])-10} more")
        print()
    
    # Create annotated video
    if args.annotate:
        print("🎨 Creating annotated video...")
        annotator = VideoAnnotator()
        output_video = output_dir / f"annotated_{video_path.name}"
        annotator.annotate_video(
            str(video_path),
            results['scores'],
            results['anomalies'],
            str(output_video),
            threshold=results['threshold']
        )
        print(f"✅ Annotated video saved: {output_video}")
        print()
    
    # Export JSON
    if args.export_json:
        print("💾 Exporting results...")
        export_data = {
            'video': str(video_path),
            'num_frames': results['num_frames'],
            'num_anomalies': results['num_anomalies'],
            'anomaly_ratio': results['anomaly_ratio'],
            'threshold': results['threshold'],
            'max_score': results['max_score'],
            'mean_score': results['mean_score'],
            'segments': results['segments'],
            'scores': results['scores'].tolist(),
            'anomalies': results['anomalies'].tolist()
        }
        
        json_path = output_dir / f"results_{video_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Results exported: {json_path}")
        print()
    
    print("=" * 70)
    print("✅ DETECTION COMPLETE!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
