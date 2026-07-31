"""Shared styling and small helpers used by every page of the app."""

import streamlit as st

PALETTE = {
    "bg": "#F6F7F5",
    "surface": "#FFFFFF",
    "ink": "#1C2321",
    "muted": "#5B655F",
    "line": "#E1E4DF",
    "teal": "#1F5E52",
    "teal_soft": "#E3EDE9",
    "amber": "#C0722C",
    "amber_soft": "#F5E7D8",
    "red": "#A3402E",
    "red_soft": "#F3E1DC",
}


def inject_base_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {PALETTE['bg']};
        }}

        h1, h2, h3 {{
            font-family: 'Space Grotesk', sans-serif !important;
            color: {PALETTE['ink']} !important;
            letter-spacing: -0.01em;
        }}

        p, li, label, .stMarkdown {{
            color: {PALETTE['ink']};
        }}

        .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {PALETTE['teal']};
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}

        .app-card {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['line']};
            border-radius: 14px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }}

        .metric-tile {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['line']};
            border-left: 4px solid {PALETTE['teal']};
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }}

        .metric-tile .value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.6rem;
            font-weight: 600;
            color: {PALETTE['ink']};
        }}

        .metric-tile .label {{
            font-size: 0.8rem;
            color: {PALETTE['muted']};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .price-tag {{
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 2.6rem;
            color: {PALETTE['teal']};
            background: {PALETTE['teal_soft']};
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            display: inline-block;
        }}

        .risk-badge {{
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 1.4rem;
            border-radius: 12px;
            padding: 0.9rem 1.3rem;
            display: inline-block;
        }}

        .risk-high {{ background: {PALETTE['red_soft']}; color: {PALETTE['red']}; }}
        .risk-low  {{ background: {PALETTE['teal_soft']}; color: {PALETTE['teal']}; }}

        .stButton>button {{
            background-color: {PALETTE['teal']};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
        }}
        .stButton>button:hover {{
            background-color: #163f37;
            color: white;
        }}

        [data-testid="stSidebar"] {{
            background-color: {PALETTE['surface']};
            border-right: 1px solid {PALETTE['line']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, subtitle: str):
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top:0;'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:{PALETTE['muted']}; font-size:1.05rem; max-width:640px;'>{subtitle}</p>",
        unsafe_allow_html=True,
    )
