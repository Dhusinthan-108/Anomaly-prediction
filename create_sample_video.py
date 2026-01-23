
import cv2
import numpy as np

def create_sample_video(filename='sample_surveillance.mp4', duration=10, fps=30):
    width, height = 640, 480
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Total frames
    num_frames = duration * fps
    
    print(f"Generating {filename} ({duration}s, {fps}fps)...")
    
    for i in range(num_frames):
        # Create a "normal" gray background frame (simulating static background)
        frame = np.full((height, width, 3), 50, dtype=np.uint8)
        
        # Add some static noise/grain to look like a camera
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add timestamps and static text
        cv2.putText(frame, f"CAM-01  2026-01-21  15:20:{i//fps:02d}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # ANOMALY: From frame 100 to 200 (approx 3s to 6s), an object moves across
        if 100 <= i <= 200:
            # Calculate position
            x_pos = int((i - 100) * 5) + 50
            y_pos = 240
            
            # Draw a "suspicious" red object
            cv2.circle(frame, (x_pos, y_pos), 30, (0, 0, 255), -1)
            cv2.putText(frame, "ANOMALY", (x_pos-40, y_pos-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)
        
        if i % 30 == 0:
            print(f"Generated {i/fps:.1f}s / {duration}s")

    out.release()
    print(f"✅ Saved sample video to: {filename}")

if __name__ == '__main__':
    create_sample_video()
