import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
from plotly.subplots import make_subplots




def plot_sensitivities(fly, pcs, title):
    # Create a subplot grid (3 columns, 1 row)
    fig = make_subplots(
        rows=1, cols=len(pcs.columns),  # One row, len(pcs.columns) columns
        subplot_titles=[f"{title} exposure to {b}" for b in pcs.columns],
        shared_yaxes=False,  # To share the y-axis between all subplots
        shared_xaxes=False
    )

    for idx, b in enumerate(pcs.columns):
        # Fit model (OLS regression)
        fit = sm.OLS(endog=fly, exog=pcs[b].values, missing='drop', hasconst=False).fit()
        slope = fit.params[0]
        rsq = fit.rsquared

        # Generate x-values for the regression line
        ls = np.linspace(min(pcs[b]), max(pcs[b]), 100)
        predicted_y = fit.predict(ls)

        # Scatter plot for the current variable
        fig.add_trace(go.Scatter(
            x=pcs[b].values, y=fly,
            mode='markers',
            name=f'{b} scatter',
            marker=dict(opacity=0.5)
        ), row=1, col=idx + 1)

        # Add regression line (plot as dashed line)
        fig.add_trace(go.Scatter(
            x=ls, y=predicted_y,
            mode='lines',
            name=f'{b} regression line',
            line=dict(dash='dash', color='red')
        ), row=1, col=idx + 1)

        # Add annotation with the slope (Beta)
        fig.add_annotation(
            x=ls[15], y=max(fly) - 0.2 * max(fly),
            text=f'RSq: {rsq:.3f}',
            showarrow=False,
            font=dict(size=8),
            align='left',
            row=1, col=idx + 1
        )

    # Update layout for the whole figure
    fig.update_layout(
        title=f"{title} Sensitivities",
        height=600, width=900,
        template='plotly_white',
        showlegend=False
    )

    # Show the plot
    fig.show()

    return fig

def plot_heat_map(corr_matrix,title):
    text = corr_matrix.map(lambda x: f"{x:.0%}")
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        text=text.values,  # percent strings
        hoverinfo='text',
        colorscale='RdYlGn',  # red to green
        zmin=-1, zmax=1,
        showscale=True
    ))

    # Add annotations (optional for visible percent text in cells)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                text=text.iloc[i, j],
                showarrow=False,
                font=dict(color="black", size=10)
            )

    fig.update_layout(
        title=title,
       # xaxis_title="Variables",
       # yaxis_title="Variables",
        xaxis=dict(tickangle=45),
        width=600,
        height=600
    )

    return fig

def format_weights(df):
    format_dict = {}
    num_columns = len(df.columns)
# Apply percentage format to the first 6 columns
    for i in range(min(6, num_columns)):
        format_dict[df.columns[i]] = '{:.0%}'

# Apply float format with 0 precision to the last 2 columns
    for i in range(max(6, num_columns) - 2, num_columns):
        format_dict[df.columns[i]] = '{:,.1f}'
    formatted_df = df.style.format(format_dict)
    return formatted_df


def format_returns(df):
    stats = df.describe().T[['mean', 'std']]
    stats['skew'] = df.skew()
    stats['kurtosis'] = df.kurtosis()

    # Annualized return & volatility (weekly returns assumed)
    stats['annual_return'] = df.mean() * 52
    stats['annual_volatility'] = df.std() * np.sqrt(52)

    stats['ann_sharpe_ratio'] = df.mean()/df.std() * np.sqrt(52)

    # Max drawdown calculation
    def max_drawdown(series):
        cumulative = (1 + series).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    stats['max_drawdown'] = df.apply(max_drawdown)

    # Formatting
    formatted = stats.copy()

    percent_cols = ['mean', 'std',  'annual_return', 'annual_volatility','max_drawdown']
    for col in percent_cols:
        formatted[col] = formatted[col].map(lambda x: f"{x:.2%}")

    float_cols = ['skew', 'kurtosis','ann_sharpe_ratio']
    for col in float_cols:
        formatted[col] = formatted[col].map(lambda x: f"{x:.2f}")

    # Rename columns for clarity
    formatted = formatted.rename(columns={
        'mean': 'Mean',
        'std': 'Std Dev',
        'skew': 'Skew',
        'kurtosis': 'Kurtosis',
        'annual_return': 'Annual Return',
        'annual_volatility': 'Annual Volatility',
        'ann_sharpe_ratio': 'Ann IR',
        'max_drawdown': 'MDD'

    })
    return formatted

def plot_returns(df,title):
    fig = go.Figure()
    cumulative_returns = (1 + df).cumprod() - 1


    for asset in cumulative_returns.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[asset],
            mode='lines',
            name=asset,
        ))


    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        yaxis_tickformat=".1%",
        template='plotly_white',
    )

    fig.update_yaxes(title_text="Return", tickformat=".1%")


    fig.show()
