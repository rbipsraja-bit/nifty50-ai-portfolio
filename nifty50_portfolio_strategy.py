
"""
NIFTY 50 Portfolio Strategy, Machine Learning & Streamlit Dashboard

Input:
    NIFTY50_all.csv

Run analysis:
    python nifty50_portfolio_strategy.py

Run dashboard:
    streamlit run nifty50_portfolio_strategy.py -- --dashboard

The workflow:
1. Cleans and validates the historical NIFTY-50 data.
2. Engineers momentum, volatility, liquidity and price/volume features.
3. Measures which indicators are most associated with future volatility.
4. Trains a time-aware ML model to forecast 21-trading-day forward returns.
5. Ranks stocks using predicted return + risk + liquidity.
6. Builds a maximum-Sharpe-style portfolio subject to concentration limits.
7. Provides a Streamlit dashboard for interactive stock/portfolio analysis.

IMPORTANT:
This is a research/education model, not a guarantee of maximum future return.
The portfolio uses historical relationships and ML estimates; it does not know
future market events. Validate with out-of-sample backtesting before investing.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DATA_FILE = Path("NIFTY50_all.csv")
OUTPUT_DIR = Path("portfolio_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TRADING_DAYS = 252
FORWARD_DAYS = 21
TOP_N_STOCKS = 10
MAX_WEIGHT = 0.20
MIN_WEIGHT = 0.00
RISK_FREE_RATE = 0.06


# ---------------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------------
def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP",
        "Volume", "Turnover", "Trades", "Deliverable Volume", "%Deliverble"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep normal equity series and valid observations.
    if "Series" in df.columns:
        df = df[df["Series"].eq("EQ") | df["Series"].isna()].copy()

    df = df.dropna(subset=["Date", "Symbol", "Close"])
    df = df[df["Close"] > 0].copy()
    df = df.sort_values(["Symbol", "Date"]).drop_duplicates(
        ["Symbol", "Date"], keep="last"
    )

    # Remove impossible OHLC observations.
    valid_ohlc = (
        (df["Open"] > 0) &
        (df["High"] >= df[["Open", "Low", "Close"]].max(axis=1)) &
        (df["Low"] <= df[["Open", "High", "Close"]].min(axis=1))
    )
    df = df[valid_ohlc].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
def add_features(df):
    g = df.groupby("Symbol", group_keys=False)

    # Daily return
    df["return_1d"] = g["Close"].pct_change()

    # Momentum
    df["momentum_5d"] = g["Close"].pct_change(5)
    df["momentum_21d"] = g["Close"].pct_change(21)
    df["momentum_63d"] = g["Close"].pct_change(63)

    # Rolling volatility
    df["volatility_21d"] = (
        g["return_1d"].rolling(21).std().reset_index(level=0, drop=True)
        * np.sqrt(TRADING_DAYS)
    )
    df["volatility_63d"] = (
        g["return_1d"].rolling(63).std().reset_index(level=0, drop=True)
        * np.sqrt(TRADING_DAYS)
    )

    # Downside volatility
    negative_return = df["return_1d"].clip(upper=0)
    df["downside_vol_21d"] = (
        negative_return.groupby(df["Symbol"])
        .rolling(21).std()
        .reset_index(level=0, drop=True)
        * np.sqrt(TRADING_DAYS)
    )

    # Intraday range / ATR-like measure
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Prev Close"].replace(0, np.nan)
    df["open_gap"] = (df["Open"] - df["Prev Close"]) / df["Prev Close"].replace(0, np.nan)

    prev_close = g["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr_14"] = (
        true_range.groupby(df["Symbol"]).rolling(14).mean()
        .reset_index(level=0, drop=True)
        / df["Close"]
    )

    # Moving-average relationships
    ma_20 = g["Close"].rolling(20).mean().reset_index(level=0, drop=True)
    ma_50 = g["Close"].rolling(50).mean().reset_index(level=0, drop=True)
    ma_200 = g["Close"].rolling(200).mean().reset_index(level=0, drop=True)

    df["price_vs_ma20"] = df["Close"] / ma_20 - 1
    df["price_vs_ma50"] = df["Close"] / ma_50 - 1
    df["price_vs_ma200"] = df["Close"] / ma_200 - 1

    # RSI(14)
    delta = g["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.groupby(df["Symbol"]).rolling(14).mean()
        .reset_index(level=0, drop=True)
    )
    avg_loss = (
        loss.groupby(df["Symbol"]).rolling(14).mean()
        .reset_index(level=0, drop=True)
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volume/liquidity indicators
    volume_ma20 = (
        g["Volume"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    df["volume_ratio_20d"] = df["Volume"] / volume_ma20.replace(0, np.nan)

    df["delivery_ratio"] = pd.to_numeric(
        df["%Deliverble"], errors="coerce"
    )

    # VWAP relationship
    df["close_vs_vwap"] = df["Close"] / df["VWAP"].replace(0, np.nan) - 1

    # Log turnover for scale stability
    df["log_turnover"] = np.log1p(df["Turnover"].clip(lower=0))

    # Future target: next 21 trading-day return.
    # shift(-FORWARD_DAYS) is calculated inside each stock to avoid leakage.
    df["forward_return_21d"] = (
        g["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1
    )

    # A direct future-volatility target used only for explanatory analysis.
    future_abs_return = g["return_1d"].shift(-1).abs()
    df["future_abs_return_1d"] = future_abs_return

    return df


# ---------------------------------------------------------------------
# 3. VOLATILITY EXPLANATION
# ---------------------------------------------------------------------
def volatility_analysis(df):
    features = [
        "momentum_5d", "momentum_21d", "momentum_63d",
        "volatility_21d", "volatility_63d", "downside_vol_21d",
        "intraday_range", "open_gap", "atr_14",
        "price_vs_ma20", "price_vs_ma50", "price_vs_ma200",
        "rsi_14", "volume_ratio_20d", "delivery_ratio",
        "close_vs_vwap", "log_turnover"
    ]

    work = df[features + ["future_abs_return_1d"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Rank correlation is robust to outliers and is useful for financial data.
    corr = work[features].corrwith(work["future_abs_return_1d"], method="spearman")
    result = corr.abs().sort_values(ascending=False).to_frame("abs_spearman_corr")
    result["direction"] = np.sign(corr.loc[result.index]).map(
        {1: "positive", -1: "negative", 0: "neutral"}
    )

    result.to_csv(OUTPUT_DIR / "volatility_indicators.csv")

    print("\n=== INDICATORS MOST ASSOCIATED WITH FUTURE ABSOLUTE RETURN ===")
    print(result.head(10).round(4))

    return result


# ---------------------------------------------------------------------
# 4. ML MODEL
# ---------------------------------------------------------------------
FEATURES = [
    "momentum_5d", "momentum_21d", "momentum_63d",
    "volatility_21d", "volatility_63d", "downside_vol_21d",
    "intraday_range", "open_gap", "atr_14",
    "price_vs_ma20", "price_vs_ma50", "price_vs_ma200",
    "rsi_14", "volume_ratio_20d", "delivery_ratio",
    "close_vs_vwap", "log_turnover"
]


def train_model(df):
    model_df = df[["Date", "Symbol"] + FEATURES + ["forward_return_21d"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Chronological split -- never randomly shuffle financial time series.
    dates = model_df["Date"].sort_values().unique()
    split_date = dates[int(len(dates) * 0.80)]

    train = model_df[model_df["Date"] < split_date]
    test = model_df[model_df["Date"] >= split_date]

    # Cap training size so the script remains practical on the full 235k-row file.
    if len(train) > 75000:
        train = train.sample(75000, random_state=42).sort_values("Date")

    X_train = train[FEATURES]
    y_train = train["forward_return_21d"]
    X_test = test[FEATURES]
    y_test = test["forward_return_21d"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=75,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    }

    print("\n=== OUT-OF-SAMPLE ML METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    importance.to_csv(OUTPUT_DIR / "ml_feature_importance.csv")

    # Predictions for test set
    test_pred = test[["Date", "Symbol", "forward_return_21d"]].copy()
    test_pred["predicted_return"] = pred
    test_pred.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Latest prediction for each stock.
    latest = df.sort_values(["Symbol", "Date"]).groupby("Symbol").tail(1).copy()
    latest = latest.replace([np.inf, -np.inf], np.nan)
    latest = latest.dropna(subset=FEATURES)

    latest["predicted_return_21d"] = model.predict(latest[FEATURES])

    return model, metrics, importance, latest


# ---------------------------------------------------------------------
# 5. STOCK RANKING
# ---------------------------------------------------------------------
def rank_stocks(latest):
    result = latest[
        [
            "Date", "Symbol", "Close", "predicted_return_21d",
            "volatility_21d", "downside_vol_21d", "volume_ratio_20d",
            "delivery_ratio", "momentum_21d", "momentum_63d"
        ]
    ].copy()

    # Risk-adjusted score. The small epsilon prevents division by zero.
    result["risk_adjusted_score"] = (
        result["predicted_return_21d"] /
        (result["volatility_21d"].abs() + 0.05)
    )

    # Penalize weak liquidity observations.
    result["liquidity_score"] = np.clip(
        result["volume_ratio_20d"].fillna(1), 0.25, 3.0
    )

    result["portfolio_score"] = (
        0.70 * result["risk_adjusted_score"].rank(pct=True) +
        0.20 * result["predicted_return_21d"].rank(pct=True) +
        0.10 * result["liquidity_score"].rank(pct=True)
    )

    result = result.sort_values(
        "portfolio_score", ascending=False
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_DIR / "stock_rankings.csv", index=False)

    print("\n=== TOP PROMISING STOCKS ===")
    print(
        result.head(TOP_N_STOCKS)[
            ["Symbol", "Close", "predicted_return_21d",
             "volatility_21d", "portfolio_score"]
        ].round(4).to_string(index=False)
    )

    return result


# ---------------------------------------------------------------------
# 6. PORTFOLIO CONSTRUCTION
# ---------------------------------------------------------------------
def build_portfolio(df, rankings):
    """
    Approximate maximum-Sharpe portfolio.

    Uses:
      - expected returns = ML predicted 21-day returns annualized
      - covariance = last 252 daily returns
      - long-only constraints
      - maximum 20% per stock

    scipy is optional. If unavailable, a risk-adjusted-score weighting method
    is used as a deterministic fallback.
    """
    selected = rankings.head(TOP_N_STOCKS).copy()
    symbols = selected["Symbol"].tolist()

    returns = (
        df[df["Symbol"].isin(symbols)]
        .pivot(index="Date", columns="Symbol", values="return_1d")
        .tail(TRADING_DAYS)
        .dropna(axis=1, how="all")
    )

    returns = returns.fillna(0)
    available = [s for s in symbols if s in returns.columns]

    selected = selected[selected["Symbol"].isin(available)].copy()
    selected = selected.set_index("Symbol").loc[available].reset_index()

    mu = selected["predicted_return_21d"].to_numpy() * (TRADING_DAYS / FORWARD_DAYS)
    cov = returns[available].cov().to_numpy() * TRADING_DAYS

    # Numerical stabilization.
    cov = cov + np.eye(len(available)) * 1e-6

    weights = None

    try:
        from scipy.optimize import minimize

        def objective(w):
            portfolio_return = np.dot(w, mu)
            portfolio_vol = np.sqrt(np.maximum(w @ cov @ w, 1e-12))
            sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_vol
            return -sharpe

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * len(available)
        initial = np.ones(len(available)) / len(available)

        solution = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10}
        )

        if solution.success:
            weights = solution.x

    except Exception as exc:
        print("Optimization fallback:", exc)

    if weights is None:
        # Fallback: positive risk-adjusted scores.
        scores = np.maximum(
            selected["portfolio_score"].to_numpy(), 1e-8
        )
        weights = scores / scores.sum()
        weights = np.minimum(weights, MAX_WEIGHT)

        # Re-normalize after cap.
        for _ in range(20):
            excess = 1 - weights.sum()
            if abs(excess) < 1e-8:
                break
            room = np.maximum(MAX_WEIGHT - weights, 0)
            if room.sum() <= 1e-8:
                break
            weights += excess * room / room.sum()

    selected["weight"] = weights
    selected["annualized_expected_return"] = mu
    selected["contribution"] = selected["weight"] * selected["annualized_expected_return"]

    port_return = float(np.dot(weights, mu))
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else np.nan

    selected.to_csv(OUTPUT_DIR / "recommended_portfolio.csv", index=False)

    print("\n=== RECOMMENDED PORTFOLIO ===")
    print(
        selected[
            ["Symbol", "weight", "predicted_return_21d",
             "annualized_expected_return"]
        ].round(4).to_string(index=False)
    )
    print(f"\nExpected annualized return: {port_return:.2%}")
    print(f"Annualized volatility:       {port_vol:.2%}")
    print(f"Estimated Sharpe ratio:       {sharpe:.2f}")

    return selected, {
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe
    }


# ---------------------------------------------------------------------
# 7. BACKTEST THE STRATEGY
# ---------------------------------------------------------------------
def simple_backtest(df, portfolio):
    """
    Simple historical test of equal/optimized holdings.

    This is intentionally separate from ML training. It does NOT claim to be
    a full production backtester. It provides a quick sanity check.
    """
    symbols = portfolio["Symbol"].tolist()
    weights = portfolio.set_index("Symbol")["weight"]

    price = (
        df[df["Symbol"].isin(symbols)]
        .pivot(index="Date", columns="Symbol", values="Close")
        .sort_index()
    )

    daily_returns = price.pct_change().fillna(0)

    weighted_returns = daily_returns[symbols].mul(
        [weights[s] for s in symbols], axis=1
    ).sum(axis=1)

    equity = (1 + weighted_returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1

    cagr = (
        equity.iloc[-1] ** (TRADING_DAYS / max(len(equity), 1)) - 1
        if len(equity) > 0 else np.nan
    )

    ann_vol = weighted_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (
        (weighted_returns.mean() * TRADING_DAYS - RISK_FREE_RATE) / ann_vol
        if ann_vol > 0 else np.nan
    )

    max_dd = drawdown.min()

    stats = {
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }

    pd.DataFrame([stats]).to_csv(
        OUTPUT_DIR / "backtest_summary.csv", index=False
    )

    equity.to_csv(OUTPUT_DIR / "portfolio_equity_curve.csv", header=["Equity"])

    return equity, stats


# ---------------------------------------------------------------------
# 8. DASHBOARD
# ---------------------------------------------------------------------
def run_dashboard(df, rankings, portfolio, backtest_equity):
    import streamlit as st
    import plotly.express as px

    st.set_page_config(
        page_title="NIFTY 50 AI Portfolio Dashboard",
        layout="wide"
    )

    st.title("NIFTY 50 AI Portfolio Strategy Dashboard")
    st.caption(
        "Historical analysis + ML return forecasting + risk-aware portfolio construction"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Market Overview", "Stock Analysis", "Portfolio", "Model Insights"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks", int(df["Symbol"].nunique()))
        c2.metric("Observations", f"{len(df):,}")
        c3.metric("Start", df["Date"].min().strftime("%Y-%m-%d"))
        c4.metric("End", df["Date"].max().strftime("%Y-%m-%d"))

        st.subheader("Top stocks by model score")
        st.dataframe(
            rankings.head(20).style.format({
                "Close": "{:.2f}",
                "predicted_return_21d": "{:.2%}",
                "volatility_21d": "{:.2%}",
                "portfolio_score": "{:.3f}"
            }),
            use_container_width=True
        )

        fig = px.bar(
            rankings.head(15),
            x="Symbol",
            y="portfolio_score",
            title="Top 15 model-ranked stocks"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        symbols = sorted(df["Symbol"].unique())
        selected_symbol = st.selectbox("Select stock", symbols)

        stock = df[df["Symbol"] == selected_symbol].sort_values("Date").copy()

        col1, col2, col3 = st.columns(3)
        latest = stock.iloc[-1]
        col1.metric("Latest Close", f"₹{latest['Close']:,.2f}")
        col2.metric("21D Momentum", f"{latest['momentum_21d']:.2%}")
        col3.metric("21D Volatility", f"{latest['volatility_21d']:.2%}")

        fig_price = px.line(
            stock.tail(756),
            x="Date",
            y="Close",
            title=f"{selected_symbol} price history"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        fig_vol = px.line(
            stock.tail(756),
            x="Date",
            y="volatility_21d",
            title=f"{selected_symbol} rolling 21-day annualized volatility"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab3:
        st.subheader("Recommended portfolio")
        st.dataframe(
            portfolio.style.format({
                "weight": "{:.2%}",
                "predicted_return_21d": "{:.2%}",
                "annualized_expected_return": "{:.2%}"
            }),
            use_container_width=True
        )

        p1, p2, p3 = st.columns(3)
        p1.metric("Expected Return", f"{portfolio['contribution'].sum():.2%}")
        p2.metric("Number of Holdings", len(portfolio))
        p3.metric("Maximum Weight", f"{portfolio['weight'].max():.2%}")

        fig = px.pie(
            portfolio,
            names="Symbol",
            values="weight",
            title="Portfolio allocation"
        )
        st.plotly_chart(fig, use_container_width=True)

        if backtest_equity is not None:
            equity_df = backtest_equity.rename("Portfolio Equity").reset_index()
            fig2 = px.line(
                equity_df,
                x=equity_df.columns[0],
                y="Portfolio Equity",
                title="Historical portfolio equity curve"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("What explains volatility?")

        vol_path = OUTPUT_DIR / "volatility_indicators.csv"
        imp_path = OUTPUT_DIR / "ml_feature_importance.csv"

        if vol_path.exists():
            vol = pd.read_csv(vol_path, index_col=0).head(10)
            st.write("Spearman relationship with future absolute daily return:")
            st.dataframe(vol, use_container_width=True)

        if imp_path.exists():
            imp = pd.read_csv(
                imp_path, index_col=0, names=["Feature", "Importance"]
            ).reset_index(drop=True)
            fig3 = px.bar(
                imp.head(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest feature importance"
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.info(
            "The model is a forecasting aid, not a promise of future performance. "
            "Use walk-forward backtesting, transaction costs, taxes and position "
            "limits before deploying capital."
        )


# ---------------------------------------------------------------------
# 9. MAIN PIPELINE
# ---------------------------------------------------------------------
def run_pipeline():
    print("Loading:", DATA_FILE.resolve())
    df = load_data()

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Symbols: {df['Symbol'].nunique()}")
    print(
        f"Date range: {df['Date'].min().date()} to "
        f"{df['Date'].max().date()}"
    )

    df = add_features(df)

    # Save engineered data.
    df.to_csv(OUTPUT_DIR / "engineered_data.csv", index=False)

    volatility = volatility_analysis(df)

    model, metrics, importance, latest = train_model(df)

    rankings = rank_stocks(latest)

    portfolio, portfolio_stats = build_portfolio(df, rankings)

    equity, backtest_stats = simple_backtest(df, portfolio)

    print("\n=== BACKTEST SANITY CHECK ===")
    for k, v in backtest_stats.items():
        print(f"{k}: {v:.2%}" if k != "Sharpe" else f"{k}: {v:.2f}")

    print("\nFiles created in:", OUTPUT_DIR.resolve())
    print(" - volatility_indicators.csv")
    print(" - ml_feature_importance.csv")
    print(" - stock_rankings.csv")
    print(" - recommended_portfolio.csv")
    print(" - test_predictions.csv")
    print(" - engineered_data.csv")
    print(" - backtest_summary.csv")
    print(" - portfolio_equity_curve.csv")

    return df, rankings, portfolio, equity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard instead of running the pipeline directly."
    )
    args, unknown = parser.parse_known_args()

    if args.dashboard:
        # Streamlit re-executes this file. Build/reuse outputs before dashboard.
        df, rankings, portfolio, equity = run_pipeline()
        run_dashboard(df, rankings, portfolio, equity)
    else:
        run_pipeline()
