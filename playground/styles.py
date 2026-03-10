"""Playground page styles: theme, spacing, cards, typography."""

PLAYGROUND_CSS = """
<style>
:root {
    --pg-accent: #6366f1;
    --pg-accent-hover: #4f46e5;
    --pg-bg: #0f172a;
    --pg-surface: #1e293b;
    --pg-text: #f1f5f9;
    --pg-muted: #94a3b8;
    --pg-border: #334155;
}

[data-testid="stAppViewContainer"] .pg-header {
    padding: 1.5rem 0 1rem 0;
    margin-bottom: 1rem;
    border-bottom: 2px solid #6366f1;
}
[data-testid="stAppViewContainer"] .pg-header h1 {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem !important;
}
[data-testid="stAppViewContainer"] .pg-header p {
    color: #64748b !important;
    font-size: 0.95rem;
    margin: 0 !important;
}

.pg-empty-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.pg-empty-card .pg-empty-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    opacity: 0.9;
}
.pg-empty-card .pg-empty-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.5rem;
}
.pg-empty-card .pg-empty-text {
    font-size: 0.9rem;
    color: #64748b;
    line-height: 1.5;
}

.pg-winner-badge {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: white;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
}
.pg-winner-badge .pg-winner-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.9;
}
.pg-winner-badge .pg-winner-name {
    font-size: 1.25rem;
    font-weight: 700;
}

[data-testid="stSidebar"] .stExpander summary {
    font-weight: 600 !important;
    color: #334155 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    margin-bottom: 1rem;
}
</style>
"""

PLOTLY_LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
    "title": {"font": {"size": 16, "color": "#1e293b"}, "x": 0.02, "xanchor": "left"},
    "paper_bgcolor": "rgba(255,255,255,0)",
    "plot_bgcolor": "rgba(248,250,252,0.8)",
    "margin": {"l": 56, "r": 24, "t": 56, "b": 48},
    "hoverlabel": {"bgcolor": "#1e293b", "font_size": 12},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
        "bgcolor": "rgba(255,255,255,0.8)",
        "bordercolor": "#e2e8f0",
        "borderwidth": 1,
    },
    "xaxis": {"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1"},
    "yaxis": {"gridcolor": "#e2e8f0", "zerolinecolor": "#cbd5e1"},
}

OPTIMIZER_COLORS = [
    "#6366f1",
    "#059669",
    "#dc2626",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#b91c1c",
    "#4f46e5",
    "#0d9488",
    "#ea580c",
    "#9333ea",
    "#0e7490",
]
