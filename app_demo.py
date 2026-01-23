"""
Demo UI - Works without PyTorch
Simplified version for testing the interface
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go

def create_demo_ui():
    """Create demo UI without model dependencies"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    .gr-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
    }
    """
    
    def demo_analyze(video, sensitivity):
        """Demo analysis function"""
        if video is None:
            return None, None, None, "❌ Please upload a video first!"
        
        # Simulate results
        num_frames = 100
        scores = np.random.rand(num_frames) * 0.3
        # Add some "anomalies"
        scores[20:30] = np.random.rand(10) * 0.5 + 0.5
        scores[60:70] = np.random.rand(10) * 0.5 + 0.5
        
        # Create timeline plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines',
            name='Anomaly Score',
            line=dict(color='#667eea', width=2),
            fill='tozeroy'
        ))
        fig.add_hline(y=1-sensitivity, line_dash="dash", line_color='#ef4444')
        fig.update_layout(
            title="🎯 Anomaly Detection Timeline (DEMO)",
            xaxis_title="Frame",
            yaxis_title="Anomaly Score",
            template="plotly_dark",
            height=400
        )
        
        # Stats
        anomalies = (scores > (1-sensitivity)).sum()
        stats = {
            "Total Frames": num_frames,
            "Anomalies Detected": int(anomalies),
            "Anomaly Ratio": f"{anomalies/num_frames:.2%}",
            "Mode": "DEMO MODE"
        }
        
        status = f"""
        ✅ **Demo Analysis Complete!**
        
        - **{anomalies}** anomalous frames detected
        - This is a DEMO - install PyTorch for real detection
        """
        
        return video, fig, stats, status
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    .gr-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
    }
    """
    
    with gr.Blocks(title="🎥 Anomaly Detector (DEMO)") as demo:
        gr.Markdown("""
        # 🎥 AI Surveillance Anomaly Detection System
        ### **DEMO MODE** - Install PyTorch for full functionality
        
        ---
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="📹 Upload Video")
                sensitivity = gr.Slider(0, 1, 0.5, label="🎯 Sensitivity")
                detect_btn = gr.Button("🔍 Analyze (Demo)", variant="primary")
                status = gr.Markdown("👆 Upload a video to test the UI")
            
            with gr.Column():
                output_video = gr.Video(label="✨ Output")
        
        with gr.Row():
            with gr.Column():
                plot = gr.Plot(label="📊 Timeline")
            with gr.Column():
                stats = gr.JSON(label="📈 Statistics")
        
        gr.Markdown("""
        ---
        ### 🔧 To Enable Full Functionality:
        
        1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
        2. Restart your terminal
        3. Run: `python app.py`
        
        Or use CPU-only PyTorch:
        ```bash
        pip uninstall torch torchvision
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ```
        """)
        
        detect_btn.click(
            fn=demo_analyze,
            inputs=[video_input, sensitivity],
            outputs=[output_video, plot, stats, status]
        )
    
    return demo, custom_css

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO MODE - Anomaly Detection System")
    print("=" * 70)
    print()
    print("Running in DEMO mode (PyTorch not required)")
    print("   Install PyTorch for full functionality")
    print()
    print("Launching demo interface...")
    print()
    
    demo, custom_css = create_demo_ui()
    demo.launch(server_port=7860, share=False, css=custom_css)
