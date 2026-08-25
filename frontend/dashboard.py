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
WINDOW_MAP = {"Last 3 days": 72, "Last 7 days": 168, "Last 14 days": 336, "Last 30 days": 720}

PLOT_FONT = dict(color="#E8ECF4", family="Inter, Segoe UI, sans-serif")
PLOT_AXIS = dict(gridcolor="#243044", zeroline=False, linecolor="#243044")
GOLD = "#E8B931"
TEAL_UP = "#2ECC9A"
RED_DOWN = "#F07178"
RANGE_VIOLET = "#B794F6"
RANGE_FILL = "rgba(183, 148, 246, 0.28)"

st.set_page_config(
    page_title="Crypto jumpiness forecast",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

      html, body, [class*="css"] { font-family: Inter, Segoe UI, sans-serif; }
      .block-container { padding-top: 1.15rem; padding-bottom: 2.4rem; max-width: 1180px; }
      header[data-testid="stHeader"] { background: rgba(11, 16, 32, 0.92); border-bottom: 1px solid #243044; }
      [data-testid="stSidebar"] { background: #0E1526; border-right: 1px solid #243044; }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { letter-spacing: 0.01em; }

      [data-testid="stMetric"] {
        background: #151C2E;
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 0.9rem 1.05rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      }
      [data-testid="stMetricLabel"],
      [data-testid="stMetricLabel"] p,
      [data-testid="stMetricLabel"] div {
        color: #E8ECF4 !important;
        font-weight: 700 !important;
      }
      [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #E8ECF4 !important;
      }
      [data-testid="stMetricDelta"] {
        font-weight: 700 !important;
      }

      div[data-testid="stExpander"] {
        background: #151C2E;
        border: 1px solid #243044;
        border-radius: 14px;
      }

      .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: linear-gradient(180deg, #151C2E 0%, #10182A 100%);
        border: 1px solid #243044;
        border-radius: 16px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
      }
      .brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
      }
      .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: rgba(232, 185, 49, 0.14);
        border: 1px solid rgba(232, 185, 49, 0.45);
        color: #E8B931;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .brand-title { color: #E8ECF4; font-size: 1.12rem; font-weight: 700; line-height: 1.2; }
      .brand-sub { color: #E8ECF4; font-size: 0.86rem; margin-top: 0.15rem; font-weight: 700; }
      .nav-chip {
        background: rgba(110, 168, 255, 0.12);
        border: 1px solid rgba(110, 168, 255, 0.35);
        color: #9EC4FF;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
      }

      .section-head {
        background: #151C2E;
        border: 1px solid #243044;
        border-radius: 16px;
        padding: 0.95rem 1.15rem 0.9rem;
        margin: 0.85rem 0 0.85rem;
      }
      .section-head h2 {
        color: #E8ECF4;
        font-size: 1.22rem;
        font-weight: 700;
        margin: 0 0 0.28rem 0;
      }
      .section-head p {
        color: #E8ECF4;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.5;
      }
      .section-head ul {
        margin: 0.45rem 0 0 1.15rem;
        padding: 0;
        color: #E8ECF4;
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.55;
      }
      .section-head li { margin: 0.15rem 0; }

      .note-card {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin: 0.55rem 0 0.7rem 0;
        border: 1px solid;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.14);
      }
      .note-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
      }
      .note-body {
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.5;
      }
      .note-blue {
        background: rgba(110, 168, 255, 0.10);
        border-color: rgba(110, 168, 255, 0.55);
      }
      .note-blue .note-kicker, .note-blue .note-body { color: #8FBEFF; }
      .note-yellow {
        background: rgba(232, 185, 49, 0.10);
        border-color: rgba(232, 185, 49, 0.55);
      }
      .note-yellow .note-kicker, .note-yellow .note-body { color: #F0D06A; }
      .note-red {
        background: rgba(240, 113, 120, 0.10);
        border-color: rgba(240, 113, 120, 0.55);
      }
      .note-red .note-kicker, .note-red .note-body { color: #F59AA0; }

      .howto {
        background: #151C2E;
        border: 1px solid #243044;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        color: #E8ECF4;
        font-size: 0.92rem;
        font-weight: 700;
        line-height: 1.5;
        margin-bottom: 0.7rem;
      }
      .howto ol { margin: 0.35rem 0 0 1.1rem; padding: 0; }
      .howto li { margin: 0.25rem 0; }

      .progress-note {
        color: #E8ECF4;
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.5;
        margin: 0.35rem 0 0.55rem 0;
      }
      .footer-bar {
        color: #8B97B0;
        font-size: 0.82rem;
        text-align: center;
        border: 1px solid #243044;
        border-radius: 12px;
        padding: 0.65rem 0.8rem;
        margin-top: 0.6rem;
        background: #12192A;
      }

      div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 650;
        min-height: 2.6rem;
      }
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


def note_box(title: str, body: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="note-card note-{tone}">
          <div class="note-kicker">{escape(title)}</div>
          <div class="note-body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(title: str, subtitle: str | None = None, bullets: list[str] | None = None) -> None:
    parts = [f"<h2>{escape(title)}</h2>"]
    if subtitle:
        parts.append(f"<p>{escape(subtitle)}</p>")
    if bullets:
        items = "".join(f"<li>{escape(item)}</li>" for item in bullets)
        parts.append(f"<ul>{items}</ul>")
    st.markdown(f'<div class="section-head">{"".join(parts)}</div>', unsafe_allow_html=True)


def apply_chart_layout(fig: go.Figure, height: int, y_title: str) -> go.Figure:
    title_font = dict(color="#E8ECF4", family="Inter, Segoe UI, sans-serif", size=14)
    tick_font = dict(color="#E8ECF4", family="Inter, Segoe UI, sans-serif", size=12)
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


def select_coin(symbol_key: str) -> None:
    st.session_state.view_symbol = symbol_key


if "view_symbol" not in st.session_state:
    st.session_state.view_symbol = SYMBOLS[0]
if "view_window" not in st.session_state:
    st.session_state.view_window = "Last 7 days"

st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">◈</div>
        <div>
          <div class="brand-title">Crypto jumpiness forecast</div>
          <div class="brand-sub">How bumpy the next 24 hours may be — not whether price goes up or down</div>
        </div>
      </div>
      <div class="nav-chip">Pick a coin below</div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_cols = st.columns(4)
for col, symbol_key in zip(nav_cols, SYMBOLS):
    with col:
        active = st.session_state.view_symbol == symbol_key
        st.button(
            COIN_META[symbol_key]["name"],
            type="primary" if active else "secondary",
            use_container_width=True,
            key=f"nav_{symbol_key}",
            on_click=select_coin,
            args=(symbol_key,),
        )

st.radio(
    "How much history to show on the charts",
    list(WINDOW_MAP),
    horizontal=True,
    key="view_window",
    help="This only changes the charts. It does not change the 24-hour forecast.",
)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        <div class="howto">
          <ol>
            <li>Tap a coin in the top bar.</li>
            <li>Pick how much history you want to see.</li>
            <li>Read the colored boxes first — they explain the number in plain words.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    note_box(
        "What this page does",
        "It guesses how jumpy the next 24 hours may be. It does not guess if the price will go up or down.",
        "yellow",
    )
    note_box(
        "Please read",
        "This is not financial advice. The guess can be wrong. Do not buy or sell based only on this page.",
        "red",
    )

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
    st.write("This page needs a running API, or a saved model bundle in `cloud/` or `data/` + `models/`.")
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
score_metrics = score_payload.get("metrics") or {}

section_head(
    f"{meta['name']} · next 24 hours",
    bullets=[
        f"Market: {symbol}",
        f"Based on: {as_of} hourly candle (not live).",
        f"Market feel: {mood}",
    ],
)
note_box(
    "Not a live ticker",
    "This page does not update every second. It uses the latest hourly candle we have, not live tick-by-tick prices.",
    "yellow",
)

section_head(
    "Today’s guess",
    "Four headline numbers, then a short explanation in the boxes below.",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Typical 1-day jump", f"{daily_vol:.2%}")
m2.metric("Hour-by-hour jumpiness", f"{pred_vol:.4%}")
m3.metric(f"{meta['ticker']} last price", f"{close:,.2f}" if close else "n/a")
if threshold is not None:
    gap_pct = (pred_vol / float(threshold) - 1.0) * 100.0
    m4.metric("Vs unusually-jumpy line", f"{gap_pct:+.1f}%", delta="above" if alert else "below")
else:
    m4.metric("Vs unusually-jumpy line", "n/a")

if daily_lo is not None and daily_hi is not None:
    note_box(
        "Likely range, not a promise",
        f"From past misses, a typical 1-day jump often lands between {float(daily_lo):.2%} and {float(daily_hi):.2%}. "
        "That band is wide on purpose. It is a guess of how bumpy the day may be, not a guarantee.",
        "yellow",
    )

if alert:
    note_box(
        "Head-up: unusually jumpy",
        f"The model thinks {meta['name']} may be more jumpy than on most of the last 30 days. "
        "That is a warning about bumpiness, not a call to buy or sell.",
        "red",
    )
else:
    note_box(
        "All clear vs the jumpy line",
        f"The model thinks a typical 1-day move is about {daily_vol:.1%}. "
        "That sits below this coin’s “unusually jumpy” line.",
        "blue",
    )

note_box(
    "What this number means",
    f"If this guess is right, {meta['name']}'s moves over the next day should look about as jumpy as a {daily_vol:.1%} "
    "one-day bump. This is not a prediction of the next price.",
    "blue",
)

if threshold:
    ratio = min(max(pred_vol / float(threshold), 0.0), 1.0)
    st.markdown(
        '<p class="progress-note">This bar fills up as the guess gets closer to an unusually jumpy day. '
        "A full bar means we are at or past that warning line.</p>",
        unsafe_allow_html=True,
    )
    st.progress(ratio)

section_head(
    "Price history",
    f"{meta['ticker']} hourly candles for the {window_label.lower()}. Green hours closed higher; red hours closed lower.",
)

candles = pd.DataFrame(candle_payload.get("candles", []))
if candles.empty:
    note_box("No price chart yet", "There is no candle history for this coin on this page.", "yellow")
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

section_head(
    "How jumpy it has been",
    bullets=[
        "Blue line = how jumpy the last 24 hours really were.",
        "Gold dot = today’s guess for the next 24 hours.",
        "Dashed red line = the unusually-jumpy warning line.",
    ],
)

realized = pd.DataFrame(candle_payload.get("realized_vol", []))
if realized.empty:
    note_box("No jumpiness history yet", "There is no past jumpiness series for this coin on this page.", "yellow")
else:
    realized["time"] = pd.to_datetime(realized["time"])
    vol_fig = go.Figure()
    vol_fig.add_trace(
        go.Scatter(
            x=realized["time"],
            y=realized["vol_24h_hist"],
            name="How jumpy the last 24h were",
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
            name="Today’s guess (next 24h)",
            mode="markers",
            marker=dict(size=14, color=GOLD, line=dict(width=2, color="#0B1020")),
        )
    )
    if threshold is not None:
        vol_fig.add_hline(
            y=float(threshold),
            line_dash="dash",
            line_color=RED_DOWN,
            annotation_text="unusually jumpy line",
            annotation_font_color=RED_DOWN,
        )
    apply_chart_layout(vol_fig, 420, "Hour-by-hour jumpiness")
    st.plotly_chart(vol_fig, use_container_width=True)

section_head(
    "Did past guesses match what happened?",
    "Gold line = what the model guessed. Blue line = what actually happened after 24 hours. "
    "Purple shaded area = the likely range. We can only check hours that already finished.",
)

points = pd.DataFrame(score_payload.get("points") or [])
if points.empty:
    note_box(
        "Nothing to check yet",
        "We need saved features and a loaded model before we can compare guesses with what really happened.",
        "yellow",
    )
else:
    points["time"] = pd.to_datetime(points["time"])
    s1, s2, s3, s4 = st.columns(4)
    mae = score_metrics.get("mae")
    rmse = score_metrics.get("rmse")
    coverage = score_metrics.get("coverage_95")
    n_settled = score_metrics.get("n_settled") or score_metrics.get("n")
    s1.metric("Average miss", f"{mae:.4%}" if isinstance(mae, (int, float)) else "n/a")
    s2.metric("Typical miss size", f"{rmse:.4%}" if isinstance(rmse, (int, float)) else "n/a")
    s3.metric(
        "Hours inside the range",
        f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "n/a",
        help="Share of finished hours where the real jumpiness sat inside the shaded likely range.",
    )
    s4.metric("Hours we can check", f"{int(n_settled)}" if n_settled else "n/a")

    score_fig = go.Figure()
    if points["lo"].notna().any() and points["hi"].notna().any():
        score_fig.add_trace(
            go.Scatter(
                x=points["time"],
                y=points["hi"],
                name="Range high",
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
                name="Likely range",
                mode="lines",
                line=dict(width=0, color=RANGE_VIOLET),
                fill="tonexty",
                fillcolor=RANGE_FILL,
            )
        )
    score_fig.add_trace(
        go.Scatter(
            x=points["time"],
            y=points["predicted"],
            name="Guess",
            mode="lines",
            line=dict(color=GOLD, width=2),
        )
    )
    score_fig.add_trace(
        go.Scatter(
            x=points["time"],
            y=points["actual"],
            name="What really happened",
            mode="lines",
            line=dict(color="#6C8CFF", width=2),
            connectgaps=False,
        )
    )
    apply_chart_layout(score_fig, 420, "Hour-by-hour jumpiness")
    st.plotly_chart(score_fig, use_container_width=True)
    if isinstance(coverage, (int, float)):
        pct = f"{coverage:.0%}"
        if coverage >= 0.95:
            note_box(
                "How well the range worked",
                f"The shaded range covered {pct} of the hours we can already check. We aimed for about 95%. "
                "So the range was a bit wide — extra careful, not a buy or sell tip.",
                "blue",
            )
        elif coverage >= 0.90:
            note_box(
                "How well the range worked",
                f"The shaded range covered {pct} of the hours we can already check. We aimed for about 95%. "
                "Close, but it missed a few hours. That is not a reason to trade.",
                "yellow",
            )
        else:
            note_box(
                "How well the range worked",
                f"The shaded range covered only {pct} of the hours we can already check. We aimed for about 95%. "
                "It missed more hours than we wanted. That is not a reason to trade.",
                "red",
            )

st.markdown(
    f'<div class="footer-bar">Last update {datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}</div>',
    unsafe_allow_html=True,
)
