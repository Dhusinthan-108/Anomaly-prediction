"""
Evaluation Metrics for Anomaly Detection
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_metrics(y_true, y_pred, y_scores=None):
    """
    Compute comprehensive evaluation metrics
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_scores: Anomaly scores (optional, for AUC-ROC)
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                          (metrics['precision'] + metrics['recall']) if \
                          (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Specificity and FPR
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Confusion matrix values
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    
    # AUC-ROC (if scores provided)
    if y_scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['auc_roc'] = 0.0
    
    return metrics


def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    """
    Plot ROC curve
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores
        title: Plot title
    Returns:
        fig: Plotly figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        line=dict(color='#667eea', width=3)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        height=500,
        width=600,
        showlegend=True
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig


def plot_pr_curve(y_true, y_scores, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores
        title: Plot title
    Returns:
        fig: Plotly figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    
    # PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR (AUC = {pr_auc:.3f})',
        line=dict(color='#10b981', width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark',
        height=500,
        width=600,
        showlegend=True
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix as heatmap
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        title: Plot title
    Returns:
        fig: Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = ['Normal', 'Anomaly']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_dark',
        height=400,
        width=500
    )
    
    return fig


def plot_metrics_comparison(metrics_dict, title="Model Comparison"):
    """
    Plot comparison of multiple models
    Args:
        metrics_dict: Dictionary of {model_name: metrics}
        title: Plot title
    Returns:
        fig: Plotly figure
    """
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig = go.Figure()
    
    for metric in metric_names:
        values = [metrics_dict[model].get(metric, 0) for model in models]
        fig.add_trace(go.Bar(
            name=metric.upper().replace('_', ' '),
            x=models,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        template='plotly_dark',
        height=500
    )
    
    return fig


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Generate dummy data
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    y_pred = (y_scores > 0.5).astype(int)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_scores)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Plot ROC curve
    fig_roc = plot_roc_curve(y_true, y_scores)
    fig_roc.write_html("test_roc.html")
    print("\n✅ Created ROC curve")
    
    # Plot PR curve
    fig_pr = plot_pr_curve(y_true, y_scores)
    fig_pr.write_html("test_pr.html")
    print("✅ Created PR curve")
    
    # Plot confusion matrix
    fig_cm = plot_confusion_matrix(y_true, y_pred)
    fig_cm.write_html("test_cm.html")
    print("✅ Created confusion matrix")
