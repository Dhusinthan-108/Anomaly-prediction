"""
Visualization Utilities
Beautiful plots and animations for results
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from pathlib import Path

class Visualizer:
    """
    Create stunning visualizations for anomaly detection results
    """
    def __init__(self, color_scheme=None):
        if color_scheme is None:
            self.colors = {
                'primary': '#667eea',
                'secondary': '#764ba2',
                'success': '#10b981',
                'warning': '#f59e0b',
                'danger': '#ef4444',
                'background': '#1a1a2e',
                'surface': '#16213e',
                'text': '#eaeaea',
            }
        else:
            self.colors = color_scheme
    
    def plot_anomaly_timeline(self, scores, threshold=None, anomalies=None, title="Anomaly Timeline"):
        """
        Create interactive timeline plot of anomaly scores
        Args:
            scores: Anomaly scores (seq_len,)
            threshold: Threshold line
            anomalies: Binary anomaly mask
            title: Plot title
        Returns:
            fig: Plotly figure
        """
        fig = go.Figure()
        
        # Anomaly scores
        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines',
            name='Anomaly Score',
            line=dict(color=self.colors['primary'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(102, 126, 234, 0.2)"
        ))
        
        # Threshold line
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=self.colors['danger'],
                annotation_text=f"Threshold: {threshold:.3f}",
                annotation_position="right"
            )
        
        # Highlight anomalies
        if anomalies is not None:
            anomaly_indices = np.where(anomalies > 0)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_indices,
                    y=scores[anomaly_indices],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color=self.colors['danger'],
                        size=10,
                        symbol='x'
                    )
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frame",
            yaxis_title="Anomaly Score",
            template="plotly_dark",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def plot_heatmap(self, scores, width=100, height=20, title="Anomaly Heatmap"):
        """
        Create heatmap visualization of anomaly scores
        Args:
            scores: Anomaly scores (seq_len,)
            width: Heatmap width
            height: Heatmap height
            title: Plot title
        Returns:
            fig: Plotly figure
        """
        # Reshape scores into 2D grid
        scores_2d = np.tile(scores, (height, 1))
        
        fig = go.Figure(data=go.Heatmap(
            z=scores_2d,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Score")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frame",
            yaxis_showticklabels=False,
            template="plotly_dark",
            height=200
        )
        
        return fig
    
    def plot_severity_gauge(self, score, title="Anomaly Severity"):
        """
        Create gauge chart for severity
        Args:
            score: Anomaly score (0-1)
            title: Plot title
        Returns:
            fig: Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [0, 33], 'color': self.colors['success']},
                    {'range': [33, 66], 'color': self.colors['warning']},
                    {'range': [66, 100], 'color': self.colors['danger']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def plot_distribution(self, scores, title="Score Distribution"):
        """
        Plot distribution of anomaly scores
        Args:
            scores: Anomaly scores
            title: Plot title
        Returns:
            fig: Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=50,
            name='Distribution',
            marker_color=self.colors['primary']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    def create_annotated_frame(self, frame, score, is_anomaly=False, text=None):
        """
        Annotate frame with anomaly information
        Args:
            frame: Frame image (H, W, C)
            score: Anomaly score
            is_anomaly: Whether frame is anomalous
            text: Optional text to display
        Returns:
            annotated: Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Color based on anomaly status
        if is_anomaly:
            color = (239, 68, 68)  # Red
            label = "⚠️ ANOMALY"
        else:
            color = (16, 185, 129)  # Green
            label = "✓ NORMAL"
        
        # Draw border
        thickness = 5
        cv2.rectangle(annotated, (0, 0), (w, h), color, thickness)
        
        # Draw label background
        label_text = f"{label} | Score: {score:.3f}"
        if text:
            label_text += f" | {text}"
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (10, 10), (10 + len(label_text) * 12, 50), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated, label_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw score bar
        bar_width = int(w * 0.3)
        bar_height = 20
        bar_x = w - bar_width - 20
        bar_y = 20
        
        # Background
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Score fill
        fill_width = int(bar_width * min(score, 1.0))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        return annotated
    
    def create_comparison_view(self, original, annotated):
        """
        Create side-by-side comparison
        Args:
            original: Original frame
            annotated: Annotated frame
        Returns:
            comparison: Combined image
        """
        # Ensure same size
        h, w = original.shape[:2]
        annotated = cv2.resize(annotated, (w, h))
        
        # Add labels
        orig_labeled = original.copy()
        cv2.putText(orig_labeled, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        annot_labeled = annotated.copy()
        cv2.putText(annot_labeled, "Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Concatenate
        comparison = np.hstack([orig_labeled, annot_labeled])
        
        return comparison


if __name__ == "__main__":
    # Test visualizer
    viz = Visualizer()
    
    # Generate dummy data
    scores = np.random.rand(100) * 0.5
    scores[50:60] = np.random.rand(10) * 0.5 + 0.5  # Add anomalies
    
    # Timeline plot
    fig1 = viz.plot_anomaly_timeline(scores, threshold=0.5)
    fig1.write_html("test_timeline.html")
    print("✅ Created timeline plot")
    
    # Heatmap
    fig2 = viz.plot_heatmap(scores)
    fig2.write_html("test_heatmap.html")
    print("✅ Created heatmap")
    
    # Gauge
    fig3 = viz.plot_severity_gauge(0.75)
    fig3.write_html("test_gauge.html")
    print("✅ Created gauge")
    
    # Distribution
    fig4 = viz.plot_distribution(scores)
    fig4.write_html("test_distribution.html")
    print("✅ Created distribution plot")
