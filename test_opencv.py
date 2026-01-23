"""
Test OpenCV functionality
"""
import cv2
import numpy as np

print("Testing OpenCV...")
print(f"OpenCV version: {cv2.__version__}")

# Create a test image
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Test getTextSize (correct function)
text = "Test Text"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2

try:
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    print(f"✅ cv2.getTextSize works: width={text_w}, height={text_h}")
except Exception as e:
    print(f"❌ cv2.getTextSize failed: {e}")

# Test putText
try:
    cv2.putText(img, text, (50, 50), font, font_scale, (255, 255, 255), thickness)
    print(f"✅ cv2.putText works")
except Exception as e:
    print(f"❌ cv2.putText failed: {e}")

# Test if getTextFont exists (it shouldn't)
try:
    result = cv2.getTextFont()
    print(f"⚠️ cv2.getTextFont exists (unexpected): {result}")
except AttributeError as e:
    print(f"✅ cv2.getTextFont doesn't exist (expected): {e}")

print("\nAll OpenCV tests completed!")
