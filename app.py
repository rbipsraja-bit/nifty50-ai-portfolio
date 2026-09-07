import streamlit as st

from nifty50_portfolio_strategy import (
    run_pipeline,
    run_dashboard
)

st.set_page_config(
    page_title="NIFTY 50 AI Portfolio",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_analysis():
    df, rankings, portfolio, equity = run_pipeline()
    return df, rankings, portfolio, equity


df, rankings, portfolio, equity = load_analysis()

run_dashboard(
    df,
    rankings,
    portfolio,
    equity
)
