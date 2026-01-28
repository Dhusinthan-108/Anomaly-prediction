"""
Cleanup script to remove old deep learning files
This removes PyTorch-related files that are no longer needed
"""

import os
import shutil
from pathlib import Path

print("=" * 70)
print("🧹 CLEANING UP OLD DEEP LEARNING FILES")
print("=" * 70)
print()

# Files to remove (old deep learning related)
files_to_remove = [
    # Old PyTorch model files
    "models/autoencoder.py",
    
    # Old training files
    "training/trainer.py",
    "train_model.py",
    
    # Old inference files (replaced by new system)
    "inference/detector.py",
    "inference/annotator.py",
    
    # Old UI files (using new web interface)
    "ui/components.py",
    "ui/themes.py",
    "ui/styles.css",
    "components.py",
    "themes.py",
    
    # Old demo/test files
    "app_demo.py",
    "debug_torch.py",
    "test_torch.py",
    "test_opencv.py",
    "test_visualization.py",
    
    # Old HTML files (using new index.html)
    "ai-code-editor.html",
    "model_training.html",
    "premium-frontend.html",
    
    # Old config (not needed anymore)
    "config.py",
    
    # Log files
    "app_log.txt",
    "crash_log.txt",
    "error_log.txt",
    "test_log.txt",
    
    # Old documentation (replaced by new docs)
    "QUICKSTART.md",
    "PERFORMANCE_OPTIMIZATION.md",
    
    # Duplicate CSS
    "styles.css",  # Keep style.css
    
    # Large unnecessary file
    "vc_redist.x64.exe",
    
    # Debug scripts
    "debug_import.py",
    "create_sample_video.py",
]

# Directories to remove (old deep learning related)
dirs_to_remove = [
    "models",
    "training",
    "ui",
    "checkpoints",
    "examples",
]

# Track what was removed
removed_files = []
removed_dirs = []
not_found = []

# Remove files
print("📄 Removing old files...")
for file_path in files_to_remove:
    full_path = Path(file_path)
    if full_path.exists():
        try:
            os.remove(full_path)
            removed_files.append(file_path)
            print(f"   ✅ Removed: {file_path}")
        except Exception as e:
            print(f"   ❌ Error removing {file_path}: {e}")
    else:
        not_found.append(file_path)

print()

# Remove directories
print("📁 Removing old directories...")
for dir_path in dirs_to_remove:
    full_path = Path(dir_path)
    if full_path.exists() and full_path.is_dir():
        try:
            shutil.rmtree(full_path)
            removed_dirs.append(dir_path)
            print(f"   ✅ Removed: {dir_path}/")
        except Exception as e:
            print(f"   ❌ Error removing {dir_path}: {e}")
    else:
        not_found.append(dir_path)

print()
print("=" * 70)
print("📊 CLEANUP SUMMARY")
print("=" * 70)
print(f"✅ Files removed: {len(removed_files)}")
print(f"✅ Directories removed: {len(removed_dirs)}")
print(f"⚠️  Not found (already removed): {len(not_found)}")
print()

# Show what's kept
print("=" * 70)
print("📦 KEPT FILES (New OpenCV System)")
print("=" * 70)
print()
print("Core System:")
print("  ✅ traffic_anomaly_detector.py  (Main detection engine)")
print("  ✅ server.py                    (Flask web server)")
print("  ✅ detect_anomalies.py          (CLI tool)")
print("  ✅ app.py                       (Standalone CLI)")
print()
print("Web Interface:")
print("  ✅ index.html                   (Web UI)")
print("  ✅ style.css                    (Styling)")
print("  ✅ script.js                    (Client logic)")
print("  ✅ favicon.ico                  (Icon)")
print()
print("Utilities:")
print("  ✅ inference/postprocess.py     (Post-processing)")
print("  ✅ utils/                       (Utility modules)")
print()
print("Testing:")
print("  ✅ test_system.py               (System tests)")
print("  ✅ test_new_system.py           (New system tests)")
print()
print("Documentation:")
print("  ✅ README.md                    (Main docs)")
print("  ✅ README_NEW_SYSTEM.md         (System guide)")
print("  ✅ MIGRATION.md                 (Migration guide)")
print("  ✅ QUICKSTART_NEW.md            (Quick start)")
print("  ✅ SUMMARY.md                   (Update summary)")
print("  ✅ ARCHITECTURE.md              (Architecture)")
print()
print("Data:")
print("  ✅ uploads/                     (Input videos)")
print("  ✅ outputs/                     (Processed videos)")
print("  ✅ data/                        (Training data - optional)")
print()
print("Configuration:")
print("  ✅ requirements.txt             (Dependencies)")
print("  ✅ __init__.py                  (Package init)")
print()

print("=" * 70)
print("✅ CLEANUP COMPLETE!")
print("=" * 70)
print()
print("🎉 Your system is now clean and optimized!")
print("   Only the new OpenCV-based files remain.")
print()
print("Next steps:")
print("  1. Restart the server: python server.py")
print("  2. Test the system: python test_system.py")
print("  3. Start detecting: http://localhost:5000")
print()
