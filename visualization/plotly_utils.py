import plotly.express as px
import pandas as pd
import numpy as np


def _to_html(fig):
    """Helper to convert Plotly figure to lightweight HTML string."""
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def histogram_plot(df, column, bins=20, title=None):
    """Render an interactive Histogram for a numerical feature."""
    if title is None:
        title = f"Histogram: {column}"
    
    try:
        nbins = int(bins) if bins else 20
    except (ValueError, TypeError):
        nbins = 20
        
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        color_discrete_sequence=["#3b82f6"]
    )
    fig.update_layout(
        bargap=0.1,
        xaxis_title=column,
        yaxis_title="Frequency"
    )
    return _to_html(fig)


def box_plot(df, column, category_col=None, title=None):
    """Render an interactive Boxplot for a numerical feature with optional category grouping."""
    if category_col and category_col in df.columns:
        if title is None:
            title = f"Boxplot: {column} by {category_col}"
        fig = px.box(
            df,
            x=category_col,
            y=column,
            color=category_col,
            title=title
        )
    else:
        if title is None:
            title = f"Boxplot: {column}"
        fig = px.box(
            df,
            y=column,
            title=title,
            color_discrete_sequence=["#6366f1"]
        )
    
    fig.update_layout(
        yaxis_title=column
    )
    return _to_html(fig)


def pie_chart(df, column, title=None):
    """Render an interactive Pie Chart for a categorical/discrete feature."""
    if title is None:
        title = f"Distribution of {column}"
    
    # Value counts (top 15 categories to prevent crowded legend)
    counts = df[column].astype(str).value_counts().head(15).reset_index()
    counts.columns = [column, "count"]
    
    fig = px.pie(
        counts,
        names=column,
        values="count",
        title=title,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return _to_html(fig)


def pairplot(df, numeric_cols=None, hue_col=None):
    """Render an interactive Pairplot matrix for numerical features."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    # Limit to max 5 columns for performance and readability
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]
    
    if len(numeric_cols) < 2:
        return "<p class='text-muted' style='padding: 20px; text-align: center;'>At least 2 numerical features are required for a pairplot matrix.</p>"
    
    df_subset = df[numeric_cols].copy()
    if hue_col and hue_col in df.columns:
        df_subset[hue_col] = df[hue_col]
    
    fig = px.scatter_matrix(
        df_subset,
        dimensions=numeric_cols,
        color=hue_col if hue_col and hue_col in df.columns else None,
        title="Pairplot: Relationships Between Numeric Variables"
    )
    return _to_html(fig)
