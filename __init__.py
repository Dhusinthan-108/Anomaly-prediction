"""
UI package for Gradio interface
"""
from .components import create_dashboard
from .themes import get_custom_theme
from .styles import get_custom_css

__all__ = ['create_dashboard', 'get_custom_theme', 'get_custom_css']
