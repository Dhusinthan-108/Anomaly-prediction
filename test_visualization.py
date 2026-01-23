"""
Test the visualization module directly
"""
import numpy as np
import cv2
from utils.visualization import Visualizer

print("Testing Visualizer...")

# Create visualizer
viz = Visualizer()
print("✅ Visualizer created")

# Create a test frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print("✅ Test frame created")

# Test create_annotated_frame
try:
    annotated = viz.create_annotated_frame(frame, score=0.75, is_anomaly=True, text="Test")
    print(f"✅ create_annotated_frame works! Output shape: {annotated.shape}")
except Exception as e:
    print(f"❌ create_annotated_frame failed: {e}")
    import traceback
    traceback.print_exc()

# Test create_comparison_view
try:
    comparison = viz.create_comparison_view(frame, annotated)
    print(f"✅ create_comparison_view works! Output shape: {comparison.shape}")
except Exception as e:
    print(f"❌ create_comparison_view failed: {e}")
    import traceback
    traceback.print_exc()

print("\nAll visualization tests completed!")
