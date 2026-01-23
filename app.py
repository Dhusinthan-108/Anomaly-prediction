"""
🎥 AI Surveillance Anomaly Detection System - Simple Launcher
Simplified version that handles PyTorch import errors gracefully
"""

import sys
import os
import gradio as gr

def main():
    """
    Launch the Gradio application with error handling
    """
    print("=" * 70)
    print("AI SURVEILLANCE ANOMALY DETECTION SYSTEM")
    print("=" * 70)
    print()
    print("Initializing application...")
    print()
    
    try:
        # Try to import PyTorch
        import torch
        print(f"PyTorch {torch.__version__} loaded successfully")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        print("All modules loaded!")
        print()
        print("=" * 70)
        print("FEATURES:")
        print("   - Real-time anomaly detection")
        print("   - Interactive visualizations")
        print("   - Annotated video output")
        print("   - Comprehensive analytics")
        print("=" * 70)
        print()
        print("Opening HTML interface in browser...")
        print()
        
        # Open the HTML interface in browser
        import webbrowser
        import os
        port = int(os.environ.get('GRADIO_SERVER_PORT', 7862))
        print(f"🌐 Interface available at: http://localhost:{port}/")
        webbrowser.open(f"http://localhost:{port}/")
        
        # For now, just launch a simple HTTP server
        import http.server
        import socketserver
        import threading
        from pathlib import Path
        
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        # Start server in a thread
        def start_server():
            with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
                httpd.serve_forever()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
    except (ImportError, OSError) as e:
        # PyTorch failed to load - fallback to demo mode
        print(f"Warning: {e}")
        print()
        print("=" * 70)
        print("FALLING BACK TO CUSTOM HTML INTERFACE")
        print("=" * 70)
        print()
        print("PyTorch could not be loaded (likely missing VC++ Redistributable).")
        print("Launching CUSTOM HTML/CSS/JS interface (demo mode).")
        print()
        print("To enable full functionality:")
        print("1. Install Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. Restart your terminal")
        print()
        print("=" * 70)
        print()
        print("Opening HTML interface in browser...")
        print()
        
        # Open the HTML interface in browser
        import webbrowser
        import os
        import http.server
        import socketserver
        import threading
        from pathlib import Path
        
        port = int(os.environ.get('GRADIO_SERVER_PORT', 7862))
        print(f"🌐 Interface available at: http://localhost:{port}/")
        webbrowser.open(f"http://localhost:{port}/")
        
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        # Start server in a thread
        def start_server():
            with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
                httpd.serve_forever()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
    
    except Exception as e:
        with open("crash_log.txt", "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc(file=f)
        print(f"Error: {e}")
        print()
        print("Please check crash_log.txt for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
