
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
