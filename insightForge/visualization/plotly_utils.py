import plotly.express as px
import pandas as pd


def feature_distribution(df, column):
    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    return fig.to_html(full_html=False)


def correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap",
        aspect="auto",
    )
    return fig.to_html(full_html=False)


def target_vs_feature(df, target, feature):
    fig = px.box(
        df,
        x=target,
        y=feature,
        title=f"{feature} vs {target}",
    )
    return fig.to_html(full_html=False)
import plotly.express as px
import pandas as pd


def histogram_plot(df, column):
    fig = px.histogram(
        df, x=column, nbins=30,
        title=f"Histogram of {column}"
    )
    return fig.to_html(full_html=False)


def box_plot(df, column):
    fig = px.box(
        df, y=column,
        title=f"Boxplot of {column}"
    )
    return fig.to_html(full_html=False)


def scatter_plot(df, x_col, y_col, hue_col=None):
    if hue_col:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=hue_col,
            title=f"{y_col} vs {x_col} (colored by {hue_col})"
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_col} vs {x_col}"
        )

    return fig.to_html(full_html=False)

def pie_chart(df, column):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]

    fig = px.pie(
        counts,
        names=column,
        values="count",
        title=f"Pie Chart of {column}"
    )
    return fig.to_html(full_html=False)


def correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap",
        aspect="auto"
    )
    return fig.to_html(full_html=False)


def pairplot(df, numeric_cols=None, hue_col=None):
    """Create a pairplot for visualizing relationships between numeric variables."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    # Limit to max 5 columns for performance
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]
    
    # Create a subset dataframe
    df_subset = df[numeric_cols].copy()
    
    if hue_col and hue_col in df.columns:
        df_subset[hue_col] = df[hue_col]
    
    fig = px.scatter_matrix(
        df_subset,
        dimensions=numeric_cols,
        color=hue_col if hue_col and hue_col in df.columns else None,
        title="Pairplot: Relationships Between Numeric Variables",
        hover_data=df_subset.columns.tolist()
    )
    return fig.to_html(full_html=False)
