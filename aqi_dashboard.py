import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
 
st.set_page_config(
    page_title="AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
)
 
st.title("🌫️ AQI Dashboard")
st.markdown(
    "Upload your data files to explore Air Quality Index trends, correlations, "
    "and pollutant distributions."
)
 
AQI_ZONES = [
    (0,   50,   "green",  "Good"),
    (51,  100,  "yellow", "Moderate"),
    (101, 200,  "orange", "Sensitive"),
    (201, 300,  "red",    "Poor"),
    (301, 400,  "purple", "Very Poor"),
    (401, 500,  "maroon", "Severe"),
]
 
def aqi_category(value):
    for low, high, color, label in AQI_ZONES:
        if low <= value <= high:
            return label
    return "Unknown"
 
st.sidebar.header("📂 Upload Data Files")
 
aqi_file = st.sidebar.file_uploader(
    "AQI CSV file  (Date, City, AQI)",
    type=["csv"],
    key="aqi",
)
 
pm_file = st.sidebar.file_uploader(
    "Pollutants Excel file  (PM10, PM2.5, NOX, SO2, NH3)",
    type=["xlsx", "xls"],
    key="pm",
)
 
if aqi_file is not None:
    try:
        df_raw = pd.read_csv(aqi_file)
    except Exception as e:
        st.error(f"Could not read AQI file: {e}")
        st.stop()
 
    if "Date" in df_raw.columns and "DateTime" not in df_raw.columns:
        df_raw.rename(columns={"Date": "DateTime"}, inplace=True)
 
    required_cols = {"DateTime", "City", "AQI"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        st.error(f"AQI file is missing columns: {missing}")
        st.stop()
 
    # Feature engineering 
    df = df_raw.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
 
    df["AQI_lag1"] = df["AQI"].shift(1)
    df["AQI_lag2"] = df["AQI"].shift(2)
 
    df_clean = df.dropna().reset_index(drop=True)
 
    # Summary stats
    st.header("📊 AQI Overview")
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean AQI",   f"{df_clean['AQI'].mean():.1f}")
    col2.metric("Max AQI",    f"{df_clean['AQI'].max():.0f}")
    col3.metric("Min AQI",    f"{df_clean['AQI'].min():.0f}")
    col4.metric("Std Dev",    f"{df_clean['AQI'].std():.2f}")
 
    latest_aqi = int(df_clean["AQI"].iloc[-1])
    cat = aqi_category(latest_aqi)
    st.markdown(
        f"**Latest AQI:** {latest_aqi}  →  "
        f"<span style='background:#444;padding:3px 8px;border-radius:4px'>{cat}</span>",
        unsafe_allow_html=True,
    )
 
    with st.expander("Descriptive Statistics"):
        st.dataframe(df_clean[["AQI", "AQI_lag1", "AQI_lag2"]].describe())
 
    with st.expander("Raw Data Preview"):
        st.dataframe(df_clean)
 
    st.subheader("AQI Over Time (with quality zones)")
    fig_ts = go.Figure()
 
    x0 = df_clean["DateTime"].min()
    x1 = df_clean["DateTime"].max()
    for low, high, color, label in AQI_ZONES:
        fig_ts.add_shape(
            type="rect", x0=x0, x1=x1, y0=low, y1=high,
            fillcolor=color, opacity=0.10, line=dict(width=0), layer="below"
        )
        fig_ts.add_annotation(
            x=df_clean["DateTime"].iloc[1], y=(low + high) / 2,
            text=label, showarrow=False,
            font=dict(size=9), bgcolor="white", opacity=0.7
        )
 
    fig_ts.add_trace(go.Scatter(
        x=df_clean["DateTime"], y=df_clean["AQI"],
        mode="lines+markers", name="AQI",
        line=dict(color="#1f77b4"), marker=dict(size=6)
    ))
    fig_ts.update_layout(
        xaxis_title="Date", yaxis_title="AQI",
        template="plotly_white", height=450,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_ts, use_container_width=True)
 
    # Distribution plots 
    st.subheader("AQI Feature Distributions")
    features = ["AQI", "AQI_lag1", "AQI_lag2"]
    fig_dist = make_subplots(rows=1, cols=3, subplot_titles=features)
    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for i, (feat, clr) in enumerate(zip(features, colors), 1):
        fig_dist.add_trace(
            go.Histogram(x=df_clean[feat], name=feat, marker_color=clr,
                         opacity=0.75),
            row=1, col=i
        )
    fig_dist.update_layout(
        title="Distribution of AQI & Lag Features",
        height=350, showlegend=False, template="plotly_white"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
 
    # Correlation heatmap 
    st.subheader("Correlation Heatmap")
 
    corr_matrix = df_clean[["AQI", "AQI_lag1", "AQI_lag2"]].corr()
 
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation: AQI, AQI_lag1, AQI_lag2",
    )
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True)
 
else:
    st.info("⬆️ Upload an AQI CSV file in the sidebar to get started.")
 
# Pollutants Data
if pm_file is not None:
    st.header("🧪 Pollutants Analysis")
 
    try:
        pollutants = pd.read_excel(pm_file)
    except Exception as e:
        st.error(f"Could not read Pollutants file: {e}")
        st.stop()
 
    unnamed_cols = [c for c in pollutants.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        pollutants.drop(columns=unnamed_cols, inplace=True)
 
    if "DateTime" in pollutants.columns:
        pollutants["DateTime"] = pd.to_datetime(pollutants["DateTime"])
 
    skip_cols = {"DateTime", "City"}
    pollutant_cols = [c for c in pollutants.columns if c not in skip_cols
                      and pd.api.types.is_numeric_dtype(pollutants[c])]
 
    with st.expander("Raw Pollutants Data"):
        st.dataframe(pollutants)
 
    st.subheader("Pollutants Summary Statistics")
    st.dataframe(pollutants[pollutant_cols].describe())
 
    if "DateTime" in pollutants.columns:
        st.subheader("Pollutant Trends Over Time")
        selected = st.multiselect(
            "Select pollutants to plot:",
            options=pollutant_cols,
            default=pollutant_cols[:3] if len(pollutant_cols) >= 3 else pollutant_cols,
        )
        if selected:
            fig_pm_ts = go.Figure()
            for col in selected:
                fig_pm_ts.add_trace(go.Scatter(
                    x=pollutants["DateTime"], y=pollutants[col],
                    mode="lines+markers", name=col
                ))
            fig_pm_ts.update_layout(
                xaxis_title="DateTime", yaxis_title="Concentration",
                template="plotly_white", height=420,
                xaxis=dict(tickangle=45),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_pm_ts, use_container_width=True)
 
    # Histogram grid
    st.subheader("Pollutant Distributions")
    n_cols = 3
    n_rows = int(np.ceil(len(pollutant_cols) / n_cols))
    fig_pm_hist = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=pollutant_cols
    )
    positions = [(r + 1, c + 1)
                 for r in range(n_rows) for c in range(n_cols)]
    for i, col in enumerate(pollutant_cols):
        row, col_pos = positions[i]
        fig_pm_hist.add_trace(
            go.Histogram(x=pollutants[col], name=col),
            row=row, col=col_pos
        )
    fig_pm_hist.update_layout(
        title="Histogram of Air Pollutants",
        height=350 * n_rows, showlegend=False,
        template="plotly_white"
    )
    st.plotly_chart(fig_pm_hist, use_container_width=True)
 
    # Correlation heatmap for pollutants
    if len(pollutant_cols) >= 2:
        st.subheader("Pollutants Correlation Heatmap")
        pm_corr = pollutants[pollutant_cols].corr()
        fig_pm_heat = px.imshow(
            pm_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pollutants Correlation Matrix",
        )
        fig_pm_heat.update_layout(height=400)
        st.plotly_chart(fig_pm_heat, use_container_width=True)
 
else:
    if aqi_file is not None:
        st.info("⬆️ Upload a Pollutants Excel file in the sidebar for pollutant analysis.")