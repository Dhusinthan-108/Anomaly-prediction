try:
    import torch
    print("Success: Torch imported")
    print(f"Version: {torch.__version__}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
