import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stSidebarContent"] { padding-top: 1.5rem; }
    div[data-testid="stMetric"] label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Resources ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    models = {
        "SimpleRNN (default)" : load_model("simplernn_default.keras"),
        "SimpleRNN (tuned)"   : load_model("simplernn_tuned.keras"),
        "LSTM (default)"      : load_model("lstm_default.keras"),
        "LSTM (tuned)"        : load_model("lstm_tuned.keras"),
    }
    scaler = joblib.load("scaler.pkl")
    return models, scaler


@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── Helper Functions ───────────────────────────────────────────
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y


def predict_n_days(model, last_sequence, n_days, scaler, window_size=60):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(n_days):
        input_seq   = current_seq.reshape(1, window_size, 1)
        next_scaled = model.predict(input_seq, verbose=0)
        next_price  = scaler.inverse_transform(next_scaled)[0][0]
        predictions.append(float(next_price))
        current_seq = np.append(
            current_seq[1:],
            next_scaled.reshape(1, 1),
            axis=0
        )
    return predictions


# ── Constants ──────────────────────────────────────────────────
WINDOW_SIZE = 60
SPLIT_RATIO = 0.80

PERF = {
    "SimpleRNN (default)" : {"RMSE": 42.30, "MAE": 29.04, "R2": 0.6929},
    "SimpleRNN (tuned)"   : {"RMSE": 43.12, "MAE": 24.64, "R2": 0.6809},
    "LSTM (default)"      : {"RMSE": 25.64, "MAE": 19.09, "R2": 0.8871},
    "LSTM (tuned)"        : {"RMSE": 20.54, "MAE": 13.72, "R2": 0.9275},
}

BAR_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]


# ── Load Data & Prepare ────────────────────────────────────────
models, scaler = load_resources()
df             = load_data()

split_index  = int(len(df) * SPLIT_RATIO)
train_data   = df.iloc[:split_index]
test_data    = df.iloc[split_index:]

train_scaled = scaler.fit_transform(train_data[["Close"]])
test_scaled  = scaler.transform(test_data[["Close"]])

X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)
y_test_actual  = scaler.inverse_transform(y_test.reshape(-1, 1))


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚗 TSLA Predictor")
    st.divider()

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        index=3
    )

    horizon = st.selectbox(
        "Prediction Horizon",
        [1, 5, 10],
        format_func=lambda x: f"{x} day{'s' if x > 1 else ''} ahead"
    )

    window_size = st.selectbox(
        "Window Size (days)",
        [30, 60, 90],
        index=1
    )

    run_button = st.button("🔍 Run Prediction", use_container_width=True)

    st.divider()
    st.markdown("### 📊 Model Metrics")

    p = PERF[selected_model]
    col_a, col_b = st.columns(2)
    col_a.metric("RMSE", f"{p['RMSE']:.2f}")
    col_b.metric("MAE",  f"{p['MAE']:.2f}")
    st.metric("R² Score", f"{p['R2']:.4f}")

    st.divider()
    st.caption("Built with SimpleRNN & LSTM\nDataset: TSLA Historical Prices")


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.title("📈 Tesla Stock Price Prediction")
st.markdown(
    "Deep Learning based stock price forecasting using "
    "**SimpleRNN** and **LSTM** models — trained on TSLA historical data."
)
st.divider()

# ── Top Metric Cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

last_close    = float(df["Close"].iloc[-1])
all_time_high = float(df["Close"].max())
total_days    = len(df)
price_change  = float(df["Close"].iloc[-1] - df["Close"].iloc[-2])

c1.metric("Last Close Price",   f"${last_close:.2f}",    f"{price_change:+.2f}")
c2.metric("All Time High",      f"${all_time_high:.2f}")
c3.metric("Total Trading Days", f"{total_days:,}")
c4.metric("Best Model R²",      "0.9275",                "LSTM Tuned")

st.divider()


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Price History",
    "🔮 Model Predictions",
    "🔭 Future Forecast",
    "📊 Model Comparison",
])


# ══ TAB 1 — Price History ═════════════════════════════════════
with tab1:
    st.subheader("Tesla Historical Closing Price")

    df["MA_20"]  = df["Close"].rolling(20).mean()
    df["MA_50"]  = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    show_ma = st.multiselect(
        "Overlay Moving Averages",
        ["MA 20", "MA 50", "MA 200"],
        default=["MA 20", "MA 50"]
    )

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close Price",
        line=dict(color="#1f77b4", width=1.5)
    ))
    if "MA 20" in show_ma:
        fig1.add_trace(go.Scatter(
            x=df.index, y=df["MA_20"],
            name="MA 20",
            line=dict(color="#ff7f0e", width=1.5, dash="dot")
        ))
    if "MA 50" in show_ma:
        fig1.add_trace(go.Scatter(
            x=df.index, y=df["MA_50"],
            name="MA 50",
            line=dict(color="#2ca02c", width=1.5, dash="dot")
        ))
    if "MA 200" in show_ma:
        fig1.add_trace(go.Scatter(
            x=df.index, y=df["MA_200"],
            name="MA 200",
            line=dict(color="#d62728", width=1.5, dash="dot")
        ))

    fig1.update_layout(
        height=420,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Trading Volume")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        marker_color="#1f77b4",
        opacity=0.5
    ))
    fig_vol.update_layout(
        height=200,
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=False,
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_vol, use_container_width=True)


# ══ TAB 2 — Model Predictions ════════════════════════════════
with tab2:
    st.subheader(f"Actual vs Predicted — {selected_model}")

    model       = models[selected_model]
    pred_scaled = model.predict(X_test, verbose=0)
    pred_prices = scaler.inverse_transform(pred_scaled).flatten()
    actual_flat = y_test_actual.flatten()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=actual_flat,
        name="Actual Price",
        line=dict(color="#1f77b4", width=1.5)
    ))
    fig2.add_trace(go.Scatter(
        y=pred_prices,
        name=f"{selected_model} Predicted",
        line=dict(color="#d62728", width=1.5, dash="dash")
    ))
    fig2.update_layout(
        height=420,
        xaxis_title="Time Steps",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Prediction Error (Actual − Predicted)")
    errors = actual_flat - pred_prices

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        y=errors,
        mode="lines",
        name="Error",
        line=dict(color="#ff7f0e", width=1),
        fill="tozeroy",
        fillcolor="rgba(255,127,14,0.15)"
    ))
    fig_err.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_err.update_layout(
        height=230,
        xaxis_title="Time Steps",
        yaxis_title="Error (USD)",
        showlegend=False,
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_err, use_container_width=True)


# ══ TAB 3 — Future Forecast ═══════════════════════════════════
with tab3:
    day_label = f"{horizon} Day{'s' if horizon > 1 else ''} Ahead"
    st.subheader(f"Future Price Forecast — {day_label} | {selected_model}")

    model        = models[selected_model]
    last_seq     = test_scaled[-WINDOW_SIZE:]
    future_preds = predict_n_days(model, last_seq, horizon, scaler)
    last_price   = float(y_test_actual[-1][0])

    st.markdown(f"**Last known closing price: `${last_price:.2f}`**")
    st.markdown(" ")

    # Day-by-day metric cards
    num_cols  = min(horizon, 5)
    pred_cols = st.columns(num_cols)

    for i, price in enumerate(future_preds):
        prev  = future_preds[i - 1] if i > 0 else last_price
        delta = price - prev
        pred_cols[i % num_cols].metric(
            label=f"Day {i + 1}",
            value=f"${price:.2f}",
            delta=f"{delta:+.2f}"
        )

    st.markdown(" ")

    # Forecast chart
    last_60 = scaler.inverse_transform(
        test_scaled[-WINDOW_SIZE:]
    ).flatten()

    x_hist = list(range(WINDOW_SIZE))
    x_pred = list(range(WINDOW_SIZE - 1, WINDOW_SIZE + horizon))
    y_pred = [last_60[-1]] + future_preds

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=x_hist,
        y=last_60,
        name="Last 60 days (actual)",
        line=dict(color="#1f77b4", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=x_pred,
        y=y_pred,
        name=f"{horizon}-day forecast",
        line=dict(color="#2ca02c", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle")
    ))
    fig3.add_vline(
        x=WINDOW_SIZE - 1,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast start",
        annotation_position="top right"
    )
    fig3.update_layout(
        height=400,
        xaxis_title="Time Steps",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "⚠️ Predictions are for educational purposes only. "
        "Accuracy degrades with longer horizons due to compounding "
        "errors in recursive forecasting. Not financial advice."
    )


# ══ TAB 4 — Model Comparison ══════════════════════════════════
with tab4:
    st.subheader("Model Performance Comparison")

    model_labels = [
        "SimpleRNN (default)", "SimpleRNN (tuned)",
        "LSTM (default)",      "LSTM (tuned)"
    ]
    rmse_vals = [PERF[m]["RMSE"] for m in model_labels]
    mae_vals  = [PERF[m]["MAE"]  for m in model_labels]
    r2_vals   = [PERF[m]["R2"]   for m in model_labels]

    short_labels = [
        "SimpleRNN\n(default)", "SimpleRNN\n(tuned)",
        "LSTM\n(default)",      "LSTM\n(tuned)"
    ]

    fig4 = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "RMSE — lower is better",
            "MAE — lower is better",
            "R² Score — higher is better"
        )
    )

    for vals, col in [(rmse_vals, 1), (mae_vals, 2), (r2_vals, 3)]:
        fig4.add_trace(
            go.Bar(
                x=short_labels,
                y=vals,
                marker_color=BAR_COLORS,
                showlegend=False,
                text=[f"{v:.2f}" for v in vals],
                textposition="outside"
            ),
            row=1, col=col
        )

    fig4.update_layout(
        height=430,
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Full Results Table")

    results_df = pd.DataFrame({
        "Model" : model_labels,
        "RMSE"  : rmse_vals,
        "MAE"   : mae_vals,
        "R²"    : r2_vals,
    })

    st.dataframe(
        results_df.style
            .highlight_min(subset=["RMSE", "MAE"], color="#c8e6c9")
            .highlight_max(subset=["R²"],           color="#c8e6c9")
            .format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Key Findings")

    col_l, col_r = st.columns(2)

    with col_l:
        st.success(
            "**LSTM (tuned) is the best model** with R²=0.9275, "
            "meaning it explains 92.75% of Tesla's price variance. "
            "GridSearchCV reduced RMSE by 20% compared to LSTM default."
        )
        st.info(
            "**Why LSTM beats SimpleRNN:** LSTM's gating mechanism "
            "retains long-term dependencies that SimpleRNN loses due "
            "to the vanishing gradient problem."
        )

    with col_r:
        st.warning(
            "**Limitation:** Both models struggle with sudden extreme "
            "price spikes (e.g. Tesla's late surge to $780). "
            "Purely price-based models cannot anticipate news-driven moves."
        )
        st.info(
            "**Future improvements:** Add news sentiment (NLP), "
            "macro indicators, or explore Transformer/GRU architectures "
            "for better long-range forecasting."
        )