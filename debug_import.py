
try:
    import torch
    print("PyTorch loaded successfully")
except ImportError as e:
    print(f"PyTorch import failed: {e}")

print("Attempting to import ui...")
try:
    from ui import create_dashboard
    print("UI import successful")
except Exception as e:
    print(f"UI import failed: {e}")
    import traceback
    traceback.print_exc()
