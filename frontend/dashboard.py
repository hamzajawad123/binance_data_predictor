"""Streamlit dashboard for candles, predicted 24h volatility, and high-vol alerts."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
COIN_META = {
    "BTCUSDT": {"name": "Bitcoin", "ticker": "BTC"},
    "ETHUSDT": {"name": "Ethereum", "ticker": "ETH"},
    "BNBUSDT": {"name": "BNB", "ticker": "BNB"},
    "SOLUSDT": {"name": "Solana", "ticker": "SOL"},
}
FEATURE_GUIDE = {
    "log_return": ("This hour's move", "Log return vs the previous hour. Sign shows direction."),
    "abs_return": ("Move size", "Same return with direction removed."),
    "vol_6h": ("6h jumpiness", "Std of hourly returns over the last 6 hours."),
    "vol_24h_hist": ("24h jumpiness", "Std of hourly returns over the last 24 hours (naive baseline)."),
    "vol_72h": ("3-day jumpiness", "Std of hourly returns over the last 72 hours."),
    "vol_term_structure": ("Short vs day vol", ">1 means the last few hours were wilder than the last day."),
    "candle_range": ("Candle width", "(High − low) / close for this hour."),
    "bb_width": ("Bollinger width", "How stretched recent prices are."),
    "bb_pct": ("Band position", ">1 is above the upper band; <0 is below the lower band."),
    "rsi_14": ("RSI (14)", "Momentum. Around 70+ is often stretched high; 30− stretched low."),
    "sma_ratio": ("SMA trend", "10h SMA / 50h SMA. >1 is a short-term uptrend."),
    "ema_ratio": ("EMA trend", "12h EMA / 26h EMA. >1 is a short-term uptrend."),
    "log_volume": ("Log volume", "Log of traded volume so huge hours do not dominate."),
    "volume_z": ("Volume surprise", "How unusual volume is vs the last ~20 hours. >0 is busier."),
    "trades_z": ("Trade-count surprise", "Same idea for number of trades."),
    "taker_buy_ratio": ("Buy pressure", "Share of volume that was aggressive buys. >0.5 leans buy-side."),
    "shock_volume": ("Shock × volume", "Wide candle combined with unusual volume."),
    "hour": ("Hour (UTC)", "Hour of the candle, 0–23."),
    "dow": ("Day of week", "0 = Monday … 6 = Sunday."),
    "symbol_id": ("Coin id", "BTC=0, ETH=1, BNB=2, SOL=3."),
}
WINDOW_MAP = {"Last 3 days": 72, "Last 7 days": 168, "Last 14 days": 336, "Last 30 days": 720}

PLOT_FONT = dict(color="#E8ECF4", family="Segoe UI, Inter, sans-serif")
PLOT_AXIS = dict(gridcolor="#243044", zeroline=False, linecolor="#243044")
GOLD = "#E8B931"
TEAL_UP = "#2ECC9A"
RED_DOWN = "#F07178"
FOCUS_BLUE = "#6EA8FF"
FOCUS_YELLOW = "#E8B931"
FOCUS_RED = "#F07178"

st.set_page_config(
    page_title="Crypto Vol Forecast",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1180px; }
      header[data-testid="stHeader"] { background: rgba(11, 16, 32, 0.85); }
      [data-testid="stSidebar"] { background: #0E1526; border-right: 1px solid #243044; }
      [data-testid="stMetric"] {
        background: #151C2E;
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 0.85rem 1rem;
      }
      [data-testid="stMetricLabel"] { color: #8B97B0; }
      div[data-testid="stExpander"] { background: #151C2E; border: 1px solid #243044; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=8)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_resource
def _inprocess():
    from backend.app.main import candles, get_model_info, health, score, _predict_payload

    return {
        "health": health,
        "predict": _predict_payload,
        "candles": candles,
        "score": score,
        "model_info": get_model_info,
    }


@st.cache_data(ttl=20)
def _api_reachable() -> bool:
    forced = os.getenv("VOL_SERVE", "").strip().lower()
    if forced in {"inprocess", "local", "cloud"}:
        return False
    if forced in {"api", "http"}:
        return True
    try:
        requests.get(f"{API_URL}/health", timeout=1.5)
        return True
    except requests.RequestException:
        return False


def load_payloads(symbol: str, candle_limit: int) -> tuple[dict, dict, dict, dict, dict, str]:
    if _api_reachable():
        health = api_get("/health")
        prediction = api_post("/predict", {"symbol": symbol})
        candle_payload = api_get("/candles", symbol=symbol, limit=candle_limit)
        try:
            score_payload = api_get("/score", symbol=symbol, limit=candle_limit)
        except requests.RequestException:
            score_payload = {}
        try:
            model_info = api_get("/model-info")
        except requests.RequestException:
            model_info = {}
        return health, prediction, candle_payload, score_payload, model_info, "api"
    fns = _inprocess()
    health = fns["health"]()
    prediction = fns["predict"](symbol)
    candle_payload = fns["candles"](symbol=symbol, limit=candle_limit)
    try:
        score_payload = fns["score"](symbol=symbol, limit=candle_limit)
    except Exception:
        score_payload = {}
    try:
        model_info = fns["model_info"]()
    except Exception:
        model_info = {}
    return health, prediction, candle_payload, score_payload, model_info, "inprocess"


def fmt_ts(value: str | None) -> str:
    if not value:
        return "unknown time"
    ts = pd.to_datetime(value, utc=True)
    return ts.strftime("%d %b %Y, %H:%M UTC")


def vol_label(daily: float) -> str:
    if daily < 0.012:
        return "Quiet"
    if daily < 0.022:
        return "Typical"
    if daily < 0.035:
        return "Elevated"
    return "High"


def focus_note(text: str, color: str) -> None:
    st.markdown(
        f'<p style="color:{color};font-weight:700;font-size:1.08rem;line-height:1.45;'
        f'margin:0.55rem 0 0.75rem 0;">{text}</p>',
        unsafe_allow_html=True,
    )


def apply_chart_layout(fig: go.Figure, height: int, y_title: str) -> go.Figure:
    title_font = dict(color="#E8ECF4", family="Segoe UI, Inter, sans-serif", size=14)
    tick_font = dict(color="#E8ECF4", family="Segoe UI, Inter, sans-serif", size=12)
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        margin=dict(l=16, r=16, t=24, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_xaxes(
        title=dict(text="<b>Time (UTC)</b>", font=title_font),
        tickfont=tick_font,
        **PLOT_AXIS,
        rangeslider=dict(visible=False),
    )
    fig.update_yaxes(
        title=dict(text=f"<b>{y_title}</b>", font=title_font),
        tickfont=tick_font,
        **PLOT_AXIS,
    )
    return fig


if "view_symbol" not in st.session_state:
    st.session_state.view_symbol = SYMBOLS[0]
if "view_window" not in st.session_state:
    st.session_state.view_window = "Last 7 days"

with st.sidebar:
    st.header("Controls")
    st.caption("Choose a market, then click **Update forecast** to load it.")

    labels = [f"{COIN_META[s]['name']}  ({s})" for s in SYMBOLS]
    pending_label = st.selectbox(
        "Coin",
        labels,
        index=SYMBOLS.index(st.session_state.view_symbol),
        help="The market to forecast. The page will not change until you click Update forecast.",
    )
    pending_symbol = SYMBOLS[labels.index(pending_label)]

    pending_window = st.selectbox(
        "History range",
        list(WINDOW_MAP),
        index=list(WINDOW_MAP).index(st.session_state.view_window),
        help="How many past hourly candles to draw on the charts. This does not change the forecast itself — only how much history you see.",
    )

    apply_clicked = st.button("Update forecast", type="primary", use_container_width=True)
    if apply_clicked:
        st.session_state.view_symbol = pending_symbol
        st.session_state.view_window = pending_window

    pending = pending_symbol != st.session_state.view_symbol or pending_window != st.session_state.view_window
    if pending:
        st.info("Your selection is not applied yet. Click **Update forecast**.")

    st.divider()
    focus_note("This app forecasts how jumpy the next 24 hours may be. It does not predict whether price will go up or down.", FOCUS_YELLOW)
    focus_note("Not financial advice. Forecasts can be wrong. Do not trade solely on this number.", FOCUS_RED)
    st.caption("Streamlit Community Cloud demo — may sleep after idle time. Not a 24/7 service.")

symbol = st.session_state.view_symbol
window_label = st.session_state.view_window
candle_limit = WINDOW_MAP[window_label]
meta = COIN_META[symbol]
score_payload: dict = {}
model_info: dict = {}
serve_mode = "inprocess"

try:
    health, prediction, candle_payload, score_payload, model_info, serve_mode = load_payloads(
        symbol, candle_limit
    )
except Exception as exc:
    st.title("Could not load a forecast")
    st.write("Need either a running API or a local/cloud model bundle (`cloud/` or `data/` + `models/`).")
    st.exception(exc)
    st.stop()

pred_vol = float(prediction["predicted_vol_24h"])
daily_vol = float(prediction["predicted_vol_24h_daily"])
daily_lo = prediction.get("predicted_vol_24h_daily_low")
daily_hi = prediction.get("predicted_vol_24h_daily_high")
alert = bool(prediction.get("alert"))
threshold = prediction.get("alert_threshold")
close = prediction.get("close")
as_of = fmt_ts(prediction.get("event_timestamp"))
mood = vol_label(daily_vol)
model_ok = bool(health.get("model_loaded"))
score_metrics = score_payload.get("metrics") or {}

st.title(f"{meta['name']} volatility forecast")
st.subheader(f"{symbol} · next 24 hours")
st.caption(
    f"Based on the {as_of} hourly candle. "
    f"{'Using the trained model.' if model_ok else 'Using the persistence baseline (model not loaded).'}"
)
focus_note("This is not live tick data.", FOCUS_YELLOW)

st.divider()
st.header("Forecast")
st.caption(f"Regime: **{mood}** · Alert: **{'on' if alert else 'off'}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Expected 1-day vol", f"{daily_vol:.2%}")
m2.metric("Hourly vol (model)", f"{pred_vol:.4%}")
m3.metric(f"{meta['ticker']} close", f"{close:,.2f}" if close else "n/a")
if threshold is not None:
    gap_pct = (pred_vol / float(threshold) - 1.0) * 100.0
    m4.metric("Vs alert line", f"{gap_pct:+.1f}%", delta="above" if alert else "below")
else:
    m4.metric("Vs alert line", "n/a")

if daily_lo is not None and daily_hi is not None:
    focus_note(
        f"Typical 1-day range from past forecast errors: {float(daily_lo):.2%} – {float(daily_hi):.2%}. "
        "This is a wide statistical band, not a guarantee.",
        FOCUS_YELLOW,
    )

if alert:
    focus_note(
        escape(
            prediction.get("alert_reason")
            or "Predicted volatility is at or above the 90th percentile of the last 30 days."
        ),
        FOCUS_RED,
    )
else:
    focus_note(
        f"The model expects about a {daily_vol:.1%} typical daily move. "
        "That is below this coin’s high-volatility alert line.",
        FOCUS_BLUE,
    )

focus_note(
    f"If this forecast is right, {escape(meta['name'])}'s hourly moves over the next day "
    f"should look about as jumpy as a {daily_vol:.1%} one-day standard deviation. "
    "This is not a prediction of the next price.",
    FOCUS_BLUE,
)

if threshold:
    ratio = min(max(pred_vol / float(threshold), 0.0), 1.0)
    st.caption("Predicted hourly vol relative to the 30-day 90th-percentile alert line")
    st.progress(ratio)

st.divider()
st.header("Price history")
st.caption(f"{meta['ticker']} hourly candles · {window_label.lower()}")

candles = pd.DataFrame(candle_payload.get("candles", []))
if candles.empty:
    st.info("No candles available for this coin.")
else:
    candles["time"] = pd.to_datetime(candles["time"])
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles["time"],
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name=symbol,
                increasing_line_color=TEAL_UP,
                decreasing_line_color=RED_DOWN,
                increasing_fillcolor=TEAL_UP,
                decreasing_fillcolor=RED_DOWN,
            )
        ]
    )
    apply_chart_layout(fig, 460, "Price (USDT)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("Volatility history")
st.caption("Past 24-hour realized volatility vs the model’s next-24h forecast")

realized = pd.DataFrame(candle_payload.get("realized_vol", []))
if realized.empty:
    st.info("No realized-vol history available for this coin.")
else:
    realized["time"] = pd.to_datetime(realized["time"])
    vol_fig = go.Figure()
    vol_fig.add_trace(
        go.Scatter(
            x=realized["time"],
            y=realized["vol_24h_hist"],
            name="Realized (past 24h)",
            mode="lines",
            line=dict(color="#6C8CFF", width=2),
            fill="tozeroy",
            fillcolor="rgba(108, 140, 255, 0.12)",
        )
    )
    last_time = realized["time"].iloc[-1]
    vol_fig.add_trace(
        go.Scatter(
            x=[last_time],
            y=[pred_vol],
            name="Forecast (next 24h)",
            mode="markers",
            marker=dict(size=14, color=GOLD, line=dict(width=2, color="#0B1020")),
        )
    )
    if threshold is not None:
        vol_fig.add_hline(
            y=float(threshold),
            line_dash="dash",
            line_color=RED_DOWN,
            annotation_text="alert line",
            annotation_font_color=RED_DOWN,
        )
    apply_chart_layout(vol_fig, 420, "Hourly volatility")
    st.plotly_chart(vol_fig, use_container_width=True)

st.divider()
st.header("Predicted vs actual")
st.caption(
    "The serving model scored on recent hours. Actual 24h realized vol is only known after those 24 hours settle. "
    "This is shadow scoring of the frozen artifact, not a new walk-forward."
)

points = pd.DataFrame(score_payload.get("points") or [])
if points.empty:
    st.info("No predicted-vs-actual series yet. Need features plus a loaded model.")
else:
    points["time"] = pd.to_datetime(points["time"])
    s1, s2, s3, s4 = st.columns(4)
    mae = score_metrics.get("mae")
    rmse = score_metrics.get("rmse")
    coverage = score_metrics.get("coverage_95")
    n_settled = score_metrics.get("n_settled") or score_metrics.get("n")
    s1.metric("Shadow MAE", f"{mae:.4%}" if isinstance(mae, (int, float)) else "n/a")
    s2.metric("Shadow RMSE", f"{rmse:.4%}" if isinstance(rmse, (int, float)) else "n/a")
    s3.metric(
        "Band coverage",
        f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "n/a",
        help="Share of settled hours where realized vol sat inside the ±1.96σ residual band.",
    )
    s4.metric("Settled hours", f"{int(n_settled)}" if n_settled else "n/a")

    score_fig = go.Figure()
    if points["lo"].notna().any() and points["hi"].notna().any():
        score_fig.add_trace(
            go.Scatter(
                x=points["time"],
                y=points["hi"],
                name="Band high",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        score_fig.add_trace(
            go.Scatter(
                x=points["time"],
                y=points["lo"],
                name="≈95% residual band",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(232, 185, 49, 0.14)",
            )
        )
    score_fig.add_trace(
        go.Scatter(
            x=points["time"],
            y=points["predicted"],
            name="Predicted",
            mode="lines",
            line=dict(color=GOLD, width=2),
        )
    )
    score_fig.add_trace(
        go.Scatter(
            x=points["time"],
            y=points["actual"],
            name="Actual (settled)",
            mode="lines",
            line=dict(color="#6C8CFF", width=2),
            connectgaps=False,
        )
    )
    apply_chart_layout(score_fig, 420, "Hourly volatility")
    st.plotly_chart(score_fig, use_container_width=True)
    if isinstance(coverage, (int, float)):
        band_color = FOCUS_RED if coverage < 0.90 else FOCUS_YELLOW
        focus_note(
            f"This range caught {coverage:.0%} of settled hours in this window (target 95%). "
            "A gap means the band is imperfect — not a reason to trade harder.",
            band_color,
        )

st.caption(f"Last update {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
