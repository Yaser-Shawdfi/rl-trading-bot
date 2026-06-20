"""
Streamlit Dashboard for RL Trading Bot.
4-page interactive app: Overview, Backtest Results, Live Prediction, Market Analysis.
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

from config import MODELS_DIR, INITIAL_BALANCE
from data_loader import load_data, get_feature_columns, split_data, download_data
from trading_env import TradingEnv
from agent import evaluate_agent


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="RL Trading Bot", page_icon="🤖", layout="wide")


# ─── Cache data loading ─────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    model_path = MODELS_DIR / "ppo_trading_agent"
    if model_path.with_suffix(".zip").exists():
        return PPO.load(model_path)
    return None


@st.cache_data
def get_backtest_results():
    results_path = MODELS_DIR / "backtest_results.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


df = get_data()
model = get_model()
backtest = get_backtest_results()
features = get_feature_columns(df)
train_df, test_df = split_data(df)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
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
st.sidebar.markdown("**Model:** PPO (Stable-Baselines3)")
st.sidebar.markdown(f"**Data:** {len(df):,} days")
st.sidebar.markdown(f"**Initial Capital:** ${INITIAL_BALANCE:,}")


# ─── Overview ────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 RL Trading Bot — Overview")
    st.markdown("Reinforcement Learning agent that learns to trade stocks using PPO.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Symbol", "AAPL")
    col2.metric("Data Points", f"{len(df):,}")
    col3.metric("Features", len(features))
    col4.metric("Initial Capital", f"${INITIAL_BALANCE:,}")

    if model is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Status", "✅ Trained")
        with col2:
            st.metric("Algorithm", "PPO")
    else:
        st.warning("Model not yet trained. Run `python src/agent.py` to train.")

    st.markdown("---")
    st.subheader("Price History")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="AAPL"
    )])
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Training vs Test Split")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df["close"], name="Train", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df["close"], name="Test", line=dict(color="orange")))
    fig.update_layout(height=350, title="AAPL Close Price — Train/Test Split")
    st.plotly_chart(fig, use_container_width=True)


# ─── Backtest Results ────────────────────────────────────────────────────────
elif page == "📈 Backtest Results":
    st.title("📈 Backtest Results")

    if backtest is not None:
        st.subheader("Performance Summary")

        final_nw = backtest["net_worth"].iloc[-1]
        total_return = (final_nw - INITIAL_BALANCE) / INITIAL_BALANCE
        n_trades = int(backtest["action"].ne(backtest["action"].shift()).sum())

        # Buy & hold
        first_price = float(test_df.iloc[10]["close"])
        last_price = float(test_df.iloc[-1]["close"])
        bh_return = (last_price - first_price) / first_price

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Portfolio", f"${final_nw:,.2f}")
        col2.metric("Agent Return", f"{total_return:.2%}")
        col3.metric("Buy & Hold Return", f"{bh_return:.2%}")
        delta_color = "inverse" if total_return > bh_return else "normal"
        col4.metric("Agent vs B&H", f"{total_return - bh_return:+.2%}")

        st.markdown("---")
        st.subheader("Portfolio Value Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=backtest.index, y=backtest["net_worth"],
                                 name="Agent Portfolio", line=dict(color="blue", width=2)))

        # Buy & hold benchmark
        bh_values = [INITIAL_BALANCE]
        for i in range(1, len(backtest)):
            daily_return = (backtest["price"].iloc[i] - backtest["price"].iloc[i-1]) / backtest["price"].iloc[i-1]
            bh_values.append(bh_values[-1] * (1 + daily_return))
        fig.add_trace(go.Scatter(x=backtest.index, y=bh_values,
                                 name="Buy & Hold", line=dict(color="gray", dash="dash")))
        fig.update_layout(height=500, title="Agent vs Buy & Hold",
                          xaxis_title="Step", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trading Actions")
            action_counts = backtest["action"].value_counts().rename({0: "SELL", 1: "HOLD", 2: "BUY"})
            fig = px.bar(x=action_counts.index, y=action_counts.values,
                         color=action_counts.index,
                         color_discrete_map={"BUY": "#2ecc71", "HOLD": "#f39c12", "SELL": "#e74c3c"},
                         labels={"x": "Action", "y": "Count"})
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Price & Actions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest.index, y=backtest["price"],
                                     name="Price", line=dict(color="black", width=1)))
            buys = backtest[backtest["action"] == 2]
            sells = backtest[backtest["action"] == 0]
            fig.add_trace(go.Scatter(x=buys.index, y=buys["price"], mode="markers",
                                    marker=dict(color="green", size=6, symbol="triangle-up"),
                                    name="BUY"))
            fig.add_trace(go.Scatter(x=sells.index, y=sells["price"], mode="markers",
                                    marker=dict(color="red", size=6, symbol="triangle-down"),
                                    name="SELL"))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No backtest results found. Run `python src/agent.py` to train and evaluate.")


# ─── Live Prediction ─────────────────────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction")
    st.markdown("Run the trained agent on the test data and see its decisions in real-time.")

    if model is not None:
        if st.button("🚀 Run Agent on Test Data", type="primary"):
            with st.spinner("Running agent..."):
                results_df, summary = evaluate_agent(model, test_df, features)

                st.markdown("---")
                st.subheader("Results")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Portfolio", f"${summary['final_net_worth']:,.2f}")
                col2.metric("Agent Return", f"{summary['total_return']:.2%}")
                col3.metric("Buy & Hold", f"{summary['bh_return']:.2%}")
                col4.metric("Outperformed?", "✅ Yes" if summary["outperformed"] else "❌ No")

                st.markdown("---")
                st.subheader("Portfolio vs Buy & Hold")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=results_df["net_worth"],
                                         name="Agent", line=dict(color="blue", width=2)))

                bh_values = [INITIAL_BALANCE]
                for i in range(1, len(results_df)):
                    dr = (results_df["price"].iloc[i] - results_df["price"].iloc[i-1]) / results_df["price"].iloc[i-1]
                    bh_values.append(bh_values[-1] * (1 + dr))
                fig.add_trace(go.Scatter(y=bh_values,
                                         name="Buy & Hold", line=dict(color="gray", dash="dash")))
                fig.update_layout(height=500, xaxis_title="Step", yaxis_title="Value ($)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("Action Distribution")
                st.dataframe(results_df[["step", "price", "net_worth", "action"]].head(50),
                             use_container_width=True)
    else:
        st.warning("Model not trained. Run `python src/agent.py` first.")


# ─── Market Analysis ────────────────────────────────────────────────────────
elif page == "💹 Market Analysis":
    st.title("💹 Market Analysis")
    st.markdown("Explore the market data and technical indicators used by the agent.")

    indicator = st.selectbox("Select indicator", features + ["close", "volume"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{indicator} over time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator,
                                 line=dict(color="blue", width=1)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("RSI (Relative Strength Index)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                                 line=dict(color="purple", width=1)))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlation")
    corr_cols = ["close", "returns", "rsi", "macd", "sma_10", "sma_30", "volatility", "volume_ratio"]
    available = [c for c in corr_cols if c in df.columns]
    if len(available) > 1:
        corr = df[available].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Data Sample")
    st.dataframe(df[["close", "returns", "rsi", "macd", "sma_10", "sma_30"]].tail(20),
                 use_container_width=True)