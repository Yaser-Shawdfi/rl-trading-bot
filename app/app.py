"""
Streamlit Dashboard for Improved RL Trading Bot.
4 pages: Overview, Backtest, Live Prediction, Market Analysis.
Now with: Sharpe ratio, drawdown, win rate, position heatmap.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from stable_baselines3 import PPO
import joblib

from config import MODELS_DIR, INITIAL_BALANCE, ENSEMBLE_SEEDS
from data_loader import load_data, get_feature_columns, split_data
from trading_env import TradingEnv
from agent import evaluate_ensemble


st.set_page_config(page_title="RL Trading Bot — Improved", page_icon="🤖", layout="wide")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_ensemble():
    models = []
    for seed in ENSEMBLE_SEEDS:
        path = MODELS_DIR / f"ppo_agent_seed_{seed}"
        if path.with_suffix(".zip").exists():
            models.append(PPO.load(path))
    return models if models else None


@st.cache_data
def get_backtest():
    results_path = MODELS_DIR / "backtest_results.csv"
    summary_path = MODELS_DIR / "backtest_summary.joblib"
    results = pd.read_csv(results_path) if results_path.exists() else None
    summary = joblib.load(summary_path) if summary_path.exists() else None
    return results, summary


df = get_data()
models = get_ensemble()
backtest, summary = get_backtest()
features = get_feature_columns(df)
train_df, test_df = split_data(df)

st.sidebar.title("🤖 RL Trading Bot")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📈 Backtest Results",
    "🔮 Live Prediction",
    "💹 Market Analysis",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Symbol:** AAPL (Apple)")
st.sidebar.markdown("**Model:** PPO Ensemble (3 seeds)")
st.sidebar.markdown(f"**Features:** {len(features)} indicators")
st.sidebar.markdown(f"**Action:** Continuous (0-100%)")
st.sidebar.markdown(f"**Data:** {len(df):,} days")


if page == "📊 Overview":
    st.title("📊 RL Trading Bot — Overview")
    st.markdown("Improved RL trading agent with **continuous action space**, **risk-adjusted reward**, and **3-agent ensemble**.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbol", "AAPL")
    col2.metric("Data Points", f"{len(df):,}")
    col3.metric("Features", len(features))
    col4.metric("Agents", f"{len(models) if models else 0} (ensemble)")

    if summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agent Return", f"{summary['total_return']:.2%}")
        col2.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
        col3.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}")
        col4.metric("Win Rate", f"{summary['win_rate']:.1%}")

    st.markdown("---")
    st.subheader("Price History")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="AAPL"
    )])
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Train/Test Split")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df["close"], name="Train", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df["close"], name="Test", line=dict(color="orange")))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Improvements Over v1")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**v1 (Original)**")
        st.markdown("- 15 features\n- Discrete actions (BUY/HOLD/SELL)\n- Simple P/L reward\n- Single agent\n- Return: +36.19%")
    with col2:
        st.markdown("**v2 (Improved)**")
        st.markdown(f"- 42 features\n- Continuous actions (0-100%)\n- Risk-adjusted reward (Sharpe + DD penalty)\n- 3-agent ensemble (seeds: {ENSEMBLE_SEEDS})\n- Return: {summary['total_return']:.2%}" if summary else "- 42 features\n- Continuous actions\n- Risk-adjusted reward\n- 3-agent ensemble")


elif page == "📈 Backtest Results":
    st.title("📈 Backtest Results")

    if backtest is not None and summary is not None:
        st.subheader("Performance Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Portfolio", f"${summary['final_net_worth']:,.2f}")
        col2.metric("Agent Return", f"{summary['total_return']:.2%}")
        col3.metric("Buy & Hold", f"{summary['bh_return']:.2%}")
        col4.metric("Outperformed?", "✅ Yes" if summary["outperformed"] else "❌ No")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
        col2.metric("Max Drawdown", f"{summary['max_drawdown']:.2%}")
        col3.metric("Win Rate", f"{summary['win_rate']:.1%}")
        col4.metric("Profit Factor", f"{summary['profit_factor']:.2f}")

        st.markdown("---")
        st.subheader("Portfolio Value Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=backtest["net_worth"], name="Agent Portfolio", line=dict(color="blue", width=2)))

        bh_values = [INITIAL_BALANCE]
        for i in range(1, len(backtest)):
            dr = (backtest["price"].iloc[i] - backtest["price"].iloc[i-1]) / backtest["price"].iloc[i-1]
            bh_values.append(bh_values[-1] * (1 + dr))
        fig.add_trace(go.Scatter(y=bh_values, name="Buy & Hold", line=dict(color="gray", dash="dash")))
        fig.update_layout(height=500, xaxis_title="Step", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Position Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=backtest["position_frac"] * 100, name="Position %", fill="tozeroy", line=dict(color="green")))
            fig.update_layout(height=350, yaxis_title="% Invested")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Drawdown Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=backtest["drawdown"] * 100, name="Drawdown %", fill="tozeroy", line=dict(color="red")))
            fig.add_hline(y=0, line_color="black")
            fig.update_layout(height=350, yaxis_title="Drawdown %")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Price & Position Actions")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=backtest["price"], name="Price", line=dict(color="black", width=1)))
        colors = backtest["action"].apply(lambda x: "green" if x > 0.1 else ("red" if x < -0.1 else "gray"))
        fig.add_trace(go.Scatter(y=backtest["price"], mode="markers",
                                 marker=dict(color=colors, size=3),
                                 name="Actions"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No backtest results. Run `python src/agent.py` first.")


elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction")

    if models is not None:
        if st.button("🚀 Run Ensemble on Test Data", type="primary"):
            with st.spinner("Running 3-agent ensemble..."):
                results_df, eval_summary = evaluate_ensemble(models, test_df, features)

                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Portfolio", f"${eval_summary['final_net_worth']:,.2f}")
                col2.metric("Return", f"{eval_summary['total_return']:.2%}")
                col3.metric("Sharpe", f"{eval_summary['sharpe_ratio']:.2f}")
                col4.metric("Max DD", f"{eval_summary['max_drawdown']:.2%}")

                st.markdown("---")
                st.subheader("Portfolio vs Buy & Hold")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=results_df["net_worth"], name="Agent", line=dict(color="blue", width=2)))
                bh = [INITIAL_BALANCE]
                for i in range(1, len(results_df)):
                    dr = (results_df["price"].iloc[i] - results_df["price"].iloc[i-1]) / results_df["price"].iloc[i-1]
                    bh.append(bh[-1] * (1 + dr))
                fig.add_trace(go.Scatter(y=bh, name="Buy & Hold", line=dict(color="gray", dash="dash")))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(results_df[["step", "price", "net_worth", "position_frac", "action"]].head(50),
                             use_container_width=True)
    else:
        st.warning("Models not trained. Run `python src/agent.py` first.")


elif page == "💹 Market Analysis":
    st.title("💹 Market Analysis")

    indicator = st.selectbox("Select indicator", features + ["close", "volume"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{indicator} over time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator, line=dict(color="blue", width=1)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("RSI")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="purple")))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlation")
    corr_cols = ["close", "returns", "rsi", "macd", "stoch_k", "atr_pct", "bb_position", "volume_ratio", "momentum_10"]
    available = [c for c in corr_cols if c in df.columns]
    if len(available) > 1:
        corr = df[available].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Data Sample")
    display_cols = [c for c in ["close", "returns", "rsi", "macd", "stoch_k", "atr_pct", "bb_position", "momentum_10"] if c in df.columns]
    st.dataframe(df[display_cols].tail(20), use_container_width=True)