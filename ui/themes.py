"""
Custom Gradio Theme
"""
import gradio as gr

def get_custom_theme():
    """
    Create custom Gradio theme with professional color scheme
    """
    theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        # Colors
        body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        body_background_fill_dark="linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
        
        # Buttons
        button_primary_background_fill="#667eea",
        button_primary_background_fill_hover="#764ba2",
        button_primary_text_color="white",
        button_primary_border_color="#667eea",
        
        # Inputs
        input_background_fill="#16213e",
        input_background_fill_dark="#1a1a2e",
        input_border_color="#667eea",
        input_border_color_focus="#764ba2",
        
        # Blocks
        block_background_fill="rgba(22, 33, 62, 0.95)",
        block_border_color="rgba(102, 126, 234, 0.3)",
        block_border_width="1px",
        block_radius="16px",
        block_shadow="0 8px 32px rgba(0, 0, 0, 0.3)",
        
        # Text
        body_text_color="#eaeaea",
        body_text_color_subdued="#a0a0a0",
        
        # Borders
        border_color_primary="#667eea",
        
        # Shadows
        shadow_drop="0 4px 16px rgba(102, 126, 234, 0.4)",
        shadow_drop_lg="0 8px 32px rgba(102, 126, 234, 0.5)",
    )
    
    return theme
