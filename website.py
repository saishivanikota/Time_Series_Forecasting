import calendar
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Gold Price Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_TITLE = "Gold Price Forecasting"
DATA_PATH = "Gold Price.csv"

PRIMARY_COLOR = "#FFD700"
SECONDARY_COLOR = "#007BFF"
BACKGROUND_COLOR = "#FFFFFF"
CARD_COLOR = "#F8F9FA"
TEXT_COLOR = "#000000"
PULSE_COLOR = "#28A745"
ALERT_COLOR = "#DC3545"


def apply_styles():
    st.markdown(
        f"""
        <style>
        .reportview-container, .main, .block-container {{
            background: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
        }}
        .streamlit-expanderHeader {{ color: {TEXT_COLOR}; }}
        .stButton>button {{ border-radius: 12px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"] )
    df = df.sort_values("Date").reset_index(drop=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["MA7"] = df["Price"].rolling(window=7).mean()
    return df


@st.cache_data
def fit_arima(series_values):
    model = ARIMA(series_values, order=(5, 1, 0))
    fitted = model.fit()
    return fitted


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def get_model_confidence(mae: float, rmse: float) -> tuple[str, str]:
    """Calculate model confidence based on MAE and RMSE metrics."""
    if mae < 500 and rmse < 600:
        return "High Confidence", "🟢"
    elif mae < 1000 and rmse < 1200:
        return "Moderate Accuracy", "🟡"
    else:
        return "Fair Estimate", "🟠"


def get_did_you_know() -> list[str]:
    """Return interesting facts about gold."""
    facts = [
        "Gold prices peaked in 2020 due to global uncertainty and economic stimulus.",
        "The world's largest gold reserves are found in Fort Knox, USA.",
        "Gold is the most ductile and malleable of all metals.",
        "About 197,576 tonnes of gold have been mined throughout history.",
        "Gold doesn't corrode or tarnish, making it valuable for jewelry and coins.",
        "The price of gold is often inversely correlated with stock market performance.",
        "Central banks hold gold as a reserve asset to support their currency value.",
        "Gold mining produces approximately 3,000 tonnes of gold annually.",
    ]
    return facts


def make_insight_line(forecast_series: pd.Series) -> str:
    change = (forecast_series.iloc[-1] - forecast_series.iloc[0]) / forecast_series.iloc[0]
    direction = "stable"
    if change > 0.005:
        direction = "upward"
    elif change < -0.005:
        direction = "downward"

    volatility = np.std(np.diff(forecast_series)) / forecast_series.mean()
    if volatility < 0.004:
        volatility_text = "low volatility"
    elif volatility < 0.008:
        volatility_text = "moderate volatility"
    else:
        volatility_text = "higher volatility"

    return f"Model predicts {direction} movement with {volatility_text} over the next {len(forecast_series)} days."


def build_price_chart(df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Price"],
            mode="lines",
            name="Gold Price",
            line=dict(color="#000000", width=2),
            hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}",
        )
    )

    if forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Forecast"],
                mode="lines",
                name="Forecast",
                line=dict(color=SECONDARY_COLOR, width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"].tolist() + forecast_df["Date"].tolist()[::-1],
                y=forecast_df["Upper"].tolist() + forecast_df["Lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0, 123, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=8, r=8, t=30, b=8),
        legend=dict(bgcolor=BACKGROUND_COLOR, bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", zeroline=False),
    )

    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def build_explore_chart(df: pd.DataFrame, show_ma: bool) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Price"],
            name="Price",
            line=dict(color="#000000", width=2),
        )
    )
    if show_ma:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MA7"],
                name="7-day MA",
                line=dict(color=PRIMARY_COLOR, width=2, dash="dot"),
            )
        )

    fig.update_layout(
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=8, r=8, t=30, b=8),
        legend=dict(bgcolor=BACKGROUND_COLOR, bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, rangeselector=dict(buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]), rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def compute_metrics(series: pd.Series, holdout_days: int = 30) -> tuple[float, float]:
    if len(series) <= holdout_days + 10:
        return 0.0, 0.0
    train = series.iloc[:-holdout_days].reset_index(drop=True)
    test = series.iloc[-holdout_days:].reset_index(drop=True)
    fitted = fit_arima(train)
    forecast = fitted.predict(start=len(train), end=len(train) + holdout_days - 1, typ="levels")
    mae = float(np.mean(np.abs(forecast.values - test.values)))
    rmse = float(np.sqrt(np.mean((forecast.values - test.values) ** 2)))
    return mae, rmse


def forecast_series(df: pd.DataFrame, days: int, start_date: pd.Timestamp) -> pd.DataFrame:
    # Filter data up to start_date
    filtered_df = df[df["Date"] <= start_date].copy()
    if len(filtered_df) < 10:
        # Not enough data, use all data
        filtered_df = df.copy()
    series = filtered_df["Price"].reset_index(drop=True)
    fitted = fit_arima(series)
    forecast_obj = fitted.get_forecast(steps=days)
    dates = pd.date_range(start=start_date + timedelta(days=1), periods=days, freq="D")
    forecast_df = pd.DataFrame(
        {
            "Date": dates,
            "Forecast": forecast_obj.predicted_mean,
            "Lower": forecast_obj.conf_int(alpha=0.05).iloc[:, 0],
            "Upper": forecast_obj.conf_int(alpha=0.05).iloc[:, 1],
        }
    )
    return forecast_df


def build_dashboard(df: pd.DataFrame) -> None:
    st.markdown(f"# {PAGE_TITLE}")
    st.markdown("#### Turn gold price trends into actionable forecasts and insights.")

    forecast_toggle = st.checkbox("Show Forecast", value=True)
    forecast_df = forecast_series(df, days=7, start_date=df["Date"].max()) if forecast_toggle else None

    with st.container():
        build_price_chart(df.tail(180), forecast_df)

    current_price = float(df["Price"].iloc[-1])
    percent_change = (current_price - float(df["Price"].iloc[-8])) / float(df["Price"].iloc[-8])
    next_prediction = float(forecast_df["Forecast"].iloc[0]) if forecast_toggle else current_price
    insight_text = make_insight_line(forecast_df["Forecast"] if forecast_toggle else df["Price"].tail(7))

    trend_label = "↑" if percent_change >= 0 else "↓"
    trend_value = f"{trend_label} {abs(percent_change) * 100:.2f}%"

    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric("Current Price", format_currency(current_price), delta=None)
    col2.metric("7-day Trend", trend_value, delta=None)
    col3.metric("Next Day Prediction", format_currency(next_prediction), delta=None)

    st.markdown("---")
    
    if forecast_toggle and forecast_df is not None:
        st.markdown("### Prediction Highlight")
        pred_col1, pred_col2, pred_col3 = st.columns(3, gap="large")
        pred_col1.metric("Next Day", format_currency(next_prediction))
        pred_col2.metric("Max Predicted", format_currency(float(forecast_df["Forecast"].max())))
        pred_col3.metric("Min Predicted", format_currency(float(forecast_df["Forecast"].min())))
        st.markdown("---")

    st.markdown(f"**Insight:** {insight_text}")
    
    st.markdown("---")
    st.markdown("### Did You Know?")
    facts = get_did_you_know()
    random_fact = facts[int(current_price) % len(facts)] if facts else facts[0]
    st.info(f"💡 {random_fact}")


def build_forecast_page(df: pd.DataFrame) -> None:
    st.markdown("# Forecast")
    st.markdown("Select a start date, month and year for a realistic gold forecast scenario.")

    years = sorted(df["Date"].dt.year.unique(), reverse=True)
    selected_year = st.sidebar.selectbox("Year", years, index=0)
    months = sorted(df[df["Date"].dt.year == selected_year]["Date"].dt.month.unique())
    selected_month = st.sidebar.selectbox(
        "Month",
        months,
        format_func=lambda x: calendar.month_name[x],
        index=len(months) - 1,
    )
    month_days = sorted(
        df[(df["Date"].dt.year == selected_year) & (df["Date"].dt.month == selected_month)]["Date"].dt.day.unique()
    )
    selected_day = st.sidebar.selectbox("Day", month_days, index=len(month_days) - 1)

    start_date = pd.Timestamp(year=selected_year, month=selected_month, day=int(selected_day))
    forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1)
    model_option = st.sidebar.selectbox("Model Selection", ["ARIMA"], index=0)

    if model_option != "ARIMA":
        st.warning("Only ARIMA is available in this version.")

    forecast_df = forecast_series(df, forecast_days, start_date)
    mae, rmse = compute_metrics(df["Price"])
    summary_text = make_insight_line(forecast_df["Forecast"])

    historical_df = df[df["Date"] <= start_date]
    display_df = historical_df.tail(365)

    if display_df.empty:
        st.warning("Selected date is outside available historical data. Showing latest available range instead.")
        display_df = df.tail(365)

    with st.container():
        build_price_chart(display_df, forecast_df)

    st.markdown("---")
    st.markdown("### Forecast settings")
    st.write(f"**Start date:** {start_date.date()}")
    st.write(f"**Forecast horizon:** {forecast_days} days")

    st.markdown("---")
    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("Forecast date", forecast_df["Date"].iloc[-1].strftime("%Y-%m-%d"))

    st.markdown("---")
    confidence_label, confidence_emoji = get_model_confidence(mae, rmse)
    st.markdown("### Model Confidence")
    st.write(f"{confidence_emoji} **{confidence_label}** - Based on MAE: {mae:,.2f}, RMSE: {rmse:,.2f}")
    
    st.markdown("---")
    st.markdown("### Prediction Highlight")
    highlight_col1, highlight_col2, highlight_col3 = st.columns(3, gap="large")
    highlight_col1.metric("Next Day", format_currency(float(forecast_df["Forecast"].iloc[0])))
    highlight_col2.metric("Max Predicted", format_currency(float(forecast_df["Forecast"].max())))
    highlight_col3.metric("Min Predicted", format_currency(float(forecast_df["Forecast"].min())))
    
    st.markdown("---")
    st.markdown(f"**Forecast Summary:** {summary_text}")
    st.markdown("### Market signal updates")
    st.write(
        "- Use the date selectors to simulate real-time forecasting from your chosen point in history."
    )
    st.write("- The model refreshes instantly as you change the year, month, or day.")
    st.write("- The connected chart blends historical data with high-confidence ARIMA predictions.")


def build_explore_page(df: pd.DataFrame) -> None:
    st.markdown("# Explore")
    st.markdown("Analyze historical gold trends and choose the moving average overlay.")

    show_ma = st.checkbox("Show 7-day Moving Average", value=True)
    time_range = st.selectbox("Zoom range", ["1M", "6M", "1Y", "All"], index=3)

    if time_range == "1M":
        explore_df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=1)]
    elif time_range == "6M":
        explore_df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=6)]
    elif time_range == "1Y":
        explore_df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=1)]
    else:
        explore_df = df

    build_explore_chart(explore_df, show_ma)

    st.markdown("---")
    st.markdown("**Insight:** Gold prices show periodic fluctuations with moderate volatility. Use the zoom controls to focus on specific time windows.")
    st.markdown("**Feature:** You can now compare short-term and full-horizon trends with the same dataset, so exploration feels more like a market dashboard.")


def main() -> None:
    apply_styles()
    df = load_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Forecast", "Explore"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for clear gold analytics and interactive forecasting.")

    if page == "Dashboard":
        build_dashboard(df)
    elif page == "Forecast":
        build_forecast_page(df)
    elif page == "Explore":
        build_explore_page(df)


if __name__ == "__main__":
    main()
