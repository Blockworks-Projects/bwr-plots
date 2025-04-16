import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# --- Generate Synthetic Data ---
dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")
n_points = len(dates)

base = np.linspace(0, 10, n_points)
share_a = 0.3 + 0.1 * np.sin(base * 0.5) + np.random.rand(n_points) * 0.05
share_b = 0.4 + 0.08 * np.cos(base * 0.7) + np.random.rand(n_points) * 0.05
share_c = 0.2 - 0.05 * np.sin(base * 1.0) + np.random.rand(n_points) * 0.03
share_d = 1.0 - (share_a + share_b + share_c)
share_d = np.maximum(0, share_d + np.random.rand(n_points) * 0.02)

df_shares = pd.DataFrame(
    {
        "Product A": share_a,
        "Product B": share_b,
        "Product C": share_c,
        "Product D": share_d,
    },
    index=dates,
)
# Normalize rows to sum to 1
normalized = df_shares.div(df_shares.sum(axis=1), axis=0)

# --- Minimal Plotly Area Plot ---
fig = go.Figure()
colors = ["#8F7BE1", "#EF798A", "#779BE7", "#5637cd"]
for i, col in enumerate(normalized.columns):
    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[col],
            mode="lines",
            stackgroup="one",
            name=col,
            line=dict(width=0.5, color=colors[i % len(colors)]),
            hovertemplate="%{y:.1%}<extra>" + col + "</extra>",
        )
    )
fig.update_layout(
    title="Minimal Metric Share Area Plot",
    xaxis_title="Date",
    yaxis_title="Share",
    yaxis=dict(tickformat=".0%", range=[0, 1]),
    template="plotly_white",
    legend=dict(orientation="h"),
    width=1200,
    height=600,
)

# Save and open in browser
output_path = Path(__file__).resolve().parent.parent / "output/minimal_area_plot.html"
fig.write_html(str(output_path), auto_open=True)
print(f"Minimal area plot saved to {output_path}") 