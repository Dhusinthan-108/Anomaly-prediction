"""
Video Annotation for Anomaly Detection Results
"""
import cv2
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

from utils.visualization import Visualizer
from utils.preprocessing import VideoPreprocessor

class VideoAnnotator:
    """
    Annotate videos with anomaly detection results
    """
    def __init__(self):
        self.visualizer = Visualizer()
        self.preprocessor = VideoPreprocessor()
    
    def annotate_video(self, video_path, scores, anomalies, output_path, 
                      threshold=0.5, fps=10):
        """
        Create annotated video with anomaly detection results
        Args:
            video_path: Path to input video
            scores: Anomaly scores per frame
            anomalies: Binary anomaly mask
            output_path: Path to save annotated video
            threshold: Anomaly threshold
            fps: Output video FPS
        """
        print(f"\n🎨 Creating annotated video...")
        
        # Load video
        frames = self.preprocessor.load_video(video_path)
        
        if len(frames) != len(scores):
            print(f"⚠️ Warning: Frame count mismatch ({len(frames)} vs {len(scores)})")
            # Adjust
            min_len = min(len(frames), len(scores))
            frames = frames[:min_len]
            scores = scores[:min_len]
            anomalies = anomalies[:min_len]
        
        # Get video properties
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Annotate each frame
        for i, frame in enumerate(tqdm(frames, desc="Annotating frames")):
            score = float(scores[i])
            is_anomaly = bool(anomalies[i])
            
            # Annotate frame
            annotated = self.visualizer.create_annotated_frame(
                frame, score, is_anomaly
            )
            
            # Convert RGB to BGR for OpenCV
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(annotated_bgr)
        
        out.release()
        
        print(f"✅ Annotated video saved to: {output_path}")
    
    def create_comparison_video(self, video_path, scores, anomalies, 
                               output_path, threshold=0.5, fps=10):
        """
        Create side-by-side comparison video
        Args:
            video_path: Path to input video
            scores: Anomaly scores
            anomalies: Binary anomaly mask
            output_path: Path to save video
            threshold: Anomaly threshold
            fps: Output FPS
        """
        print(f"\n🎨 Creating comparison video...")
        
        # Load video
        frames = self.preprocessor.load_video(video_path)
        
        # Adjust lengths
        min_len = min(len(frames), len(scores))
        frames = frames[:min_len]
        scores = scores[:min_len]
        anomalies = anomalies[:min_len]
        
        # Get dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer (double width for side-by-side)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
        
        # Process frames
        for i, frame in enumerate(tqdm(frames, desc="Creating comparison")):
            score = float(scores[i])
            is_anomaly = bool(anomalies[i])
            
            # Annotate frame
            annotated = self.visualizer.create_annotated_frame(
                frame, score, is_anomaly
            )
            
            # Create comparison
            comparison = self.visualizer.create_comparison_view(frame, annotated)
            
            # Convert to BGR
            comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(comparison_bgr)
        
        out.release()
        
        print(f"✅ Comparison video saved to: {output_path}")
    
    def extract_anomaly_clips(self, video_path, segments, scores, 
                             output_dir, padding=10, fps=10):
        """
        Extract clips of detected anomalies
        Args:
            video_path: Path to input video
            segments: List of (start, end) anomaly segments
            scores: Anomaly scores
            output_dir: Directory to save clips
            padding: Frames to include before/after anomaly
            fps: Output FPS
        """
        print(f"\n✂️ Extracting anomaly clips...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video
        frames = self.preprocessor.load_video(video_path)
        
        # Extract each segment
        for idx, (start, end) in enumerate(segments):
            # Add padding
            clip_start = max(0, start - padding)
            clip_end = min(len(frames), end + padding)
            
            # Extract frames
            clip_frames = frames[clip_start:clip_end]
            clip_scores = scores[clip_start:clip_end]
            
            # Create output path
            avg_score = np.mean(clip_scores)
            output_path = output_dir / f"anomaly_{idx+1}_score_{avg_score:.3f}.mp4"
            
            # Get dimensions
            height, width = clip_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Write frames
            for i, frame in enumerate(clip_frames):
                score = float(clip_scores[i])
                is_anomaly = (i >= padding) and (i < len(clip_frames) - padding)
                
                annotated = self.visualizer.create_annotated_frame(
                    frame, score, is_anomaly,
                    text=f"Clip {idx+1}"
                )
                
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                out.write(annotated_bgr)
            
            out.release()
            
            print(f"   Saved clip {idx+1}: {output_path.name}")
        
        print(f"✅ Extracted {len(segments)} anomaly clips")


if __name__ == "__main__":
    # Test annotator
    annotator = VideoAnnotator()
    print("✅ Video annotator initialized")
