"""
Visualization module for predictions, features, and performance.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import json
import os
import sys

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not installed. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_outlook_json(path='reports/outlook.json'):
    """Load outlook JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_predictions_2d(outlook_data, output_path='reports/predictions_2d.png'):
    """Create 2D plot of predictions (probability vs volatility)."""
    df = pd.DataFrame(outlook_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by recommended action
    colors = {'Long': 'green', 'No Trade': 'orange', 'Reduce': 'red'}
    
    for action in df['recommended_action'].unique():
        subset = df[df['recommended_action'] == action]
        ax.scatter(subset['volatility'], subset['p_up_63d'], 
                  label=action, color=colors.get(action, 'gray'), 
                  s=subset['suggested_size'] * 1000, alpha=0.6)
    
    # Add threshold lines
    ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.5, label='Long threshold')
    ax.axhline(y=0.50, color='orange', linestyle='--', alpha=0.5, label='No Trade threshold')
    
    ax.set_xlabel('Volatility (Annualized)', fontsize=12)
    ax.set_ylabel('Probability of Positive Return', fontsize=12)
    ax.set_title('Stock Predictions: Probability vs Risk', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add ticker labels
    for _, row in df.iterrows():
        ax.annotate(row['ticker'], (row['volatility'], row['p_up_63d']), 
                   fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D predictions plot to {output_path}")
    plt.close()


def plot_predictions_3d(outlook_data, output_path='reports/predictions_3d.html'):
    """Create 3D plot of predictions (probability vs volatility vs size)."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping 3D plot.")
        return
    
    df = pd.DataFrame(outlook_data)
    
    colors_map = {'Long': 'green', 'No Trade': 'orange', 'Reduce': 'red'}
    df['color'] = df['recommended_action'].map(colors_map)
    
    fig = go.Figure(data=go.Scatter3d(
        x=df['volatility'],
        y=df['p_up_63d'],
        z=df['suggested_size'],
        mode='markers+text',
        marker=dict(
            size=10,
            color=df['color'],
            opacity=0.7,
            line=dict(width=1, color='black')
        ),
        text=df['ticker'],
        textposition='middle center',
        hovertemplate='<b>%{text}</b><br>' +
                      'Probability: %{y:.3f}<br>' +
                      'Volatility: %{x:.3f}<br>' +
                      'Size: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Stock Predictions: 3D View (Probability vs Volatility vs Position Size)',
        scene=dict(
            xaxis_title='Volatility',
            yaxis_title='Probability of Positive Return',
            zaxis_title='Suggested Position Size',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1200,
        height=800
    )
    
    fig.write_html(output_path)
    print(f"Saved 3D predictions plot to {output_path}")


def plot_feature_importance(feature_importance_dict, output_path='reports/feature_importance.png', top_n=20):
    """Plot feature importance."""
    if not feature_importance_dict:
        print("No feature importance data available.")
        return
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(list(feature_importance_dict.items()), columns=['feature', 'importance'])
    df = df.sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(df)), df['importance'], color='steelblue')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved feature importance plot to {output_path}")
    plt.close()


def plot_performance_over_time(validation_results, output_path='reports/performance_over_time.png'):
    """Plot model performance over time (walk-forward validation)."""
    if isinstance(validation_results, list):
        df = pd.DataFrame(validation_results)
    elif isinstance(validation_results, pd.DataFrame):
        df = validation_results.copy()
    else:
        print("Invalid validation results format.")
        return
    
    if 'test_end' not in df.columns:
        print("Missing required columns in validation results.")
        return
    
    # Convert dates
    df['test_end'] = pd.to_datetime(df['test_end'])
    df_test = df[df['split'] == 'test'].sort_values('test_end')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    metrics = ['accuracy', 'auc', 'precision', 'recall']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(df_test['test_end'], df_test[metric], marker='o', linewidth=2, markersize=6)
        ax.axhline(y=df_test[metric].mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {df_test[metric].mean():.3f}')
        ax.set_xlabel('Test Period End', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(f'{metric.upper()} Over Time', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance over time plot to {output_path}")
    plt.close()


def plot_regime_distribution(regime_data, output_path='reports/regime_distribution.png'):
    """Plot distribution of market regimes."""
    if isinstance(regime_data, pd.DataFrame):
        df = regime_data.copy()
    else:
        print("Invalid regime data format.")
        return
    
    if 'regime' not in df.columns:
        print("Missing 'regime' column in regime data.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count plot
    regime_counts = df['regime'].value_counts()
    ax1.bar(regime_counts.index, regime_counts.values, color='steelblue')
    ax1.set_xlabel('Regime', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Regime Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Time series
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        ax2.plot(df_sorted['date'], df_sorted['regime'], marker='o', markersize=3, alpha=0.6)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Regime', fontsize=12)
        ax2.set_title('Regime Over Time', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved regime distribution plot to {output_path}")
    plt.close()


def generate_all_visualizations(outlook_path='reports/outlook.json', 
                                validation_results_path=None,
                                feature_importance_path=None,
                                output_dir='reports'):
    """Generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load outlook data
    if os.path.exists(outlook_path):
        outlook_data = load_outlook_json(outlook_path)
        print("Generating prediction visualizations...")
        plot_predictions_2d(outlook_data, os.path.join(output_dir, 'predictions_2d.png'))
        plot_predictions_3d(outlook_data, os.path.join(output_dir, 'predictions_3d.html'))
    
    # Load validation results if available
    if validation_results_path and os.path.exists(validation_results_path):
        if validation_results_path.endswith('.json'):
            with open(validation_results_path, 'r') as f:
                validation_results = json.load(f)
        else:
            validation_results = pd.read_csv(validation_results_path)
        print("Generating performance visualizations...")
        plot_performance_over_time(validation_results, os.path.join(output_dir, 'performance_over_time.png'))
    
    # Load feature importance if available
    if feature_importance_path and os.path.exists(feature_importance_path):
        if feature_importance_path.endswith('.json'):
            with open(feature_importance_path, 'r') as f:
                feature_importance = json.load(f)
        else:
            feature_importance = pd.read_csv(feature_importance_path).set_index('feature')['importance'].to_dict()
        print("Generating feature importance visualization...")
        plot_feature_importance(feature_importance, os.path.join(output_dir, 'feature_importance.png'))
    
    print("All visualizations generated!")


if __name__ == '__main__':
    generate_all_visualizations()

