# streamlit_volatility_dashboard_yahoo.py
# Run: streamlit run streamlit_volatility_dashboard_yahoo.py
# Requirements:
#   pip install streamlit yfinance pandas numpy matplotlib

from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

import streamlit as st
import matplotlib.pyplot as plt

ANNUALIZATION = 252


# --- Helpers ---------------------------------------------------------------

def safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


def compute_realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Realized volatility from daily log returns, annualized."""
    logret = np.log(close).diff()
    hv = logret.rolling(window=window).std() * np.sqrt(ANNUALIZATION)
    return hv


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the last observation inside each rolling window.
    Returns values in [0,1].
    """
    def _last_rank(x):
        s = pd.Series(x)
        return s.rank(pct=True).iloc[-1]

    return series.rolling(window=window, min_periods=window).apply(_last_rank, raw=False)


def regime_from_percentile(p: float) -> tuple[str, str]:
    """Map percentile to (label, color)."""
    if np.isnan(p):
        return ("N/A", "gray")
    if p >= 0.80:
        return ("HIGH VOL", "red")
    if p >= 0.60:
        return ("ABOVE AVG", "orange")
    if p >= 0.40:
        return ("NORMAL", "black")
    if p >= 0.20:
        return ("BELOW AVG", "blue")
    return ("LOW VOL", "green")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")


def clamp_lookback_to_data(hv_short: pd.Series, requested_lookback: int) -> int:
    """
    Ensure lookback doesn't exceed the amount of available (non-NaN) HV_short values.
    Keep a sensible minimum of 30.
    """
    available = int(hv_short.dropna().shape[0])
    # Need at least 'lookback' non-NaN points for rolling percentile with min_periods=lookback.
    # Also keep it within reasonable bounds.
    lb = min(int(requested_lookback), available)
    lb = max(30, lb)
    return lb


# --- Data / Compute --------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_yahoo(symbol: str, period: str) -> pd.DataFrame:
    """
    Fetch daily adjusted OHLCV data from Yahoo via yfinance.

    Notes:
      - yfinance can intermittently throttle / fail on Streamlit Cloud.
      - If you see errors, retry, use a different period, or wait a bit.
    """
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance can return MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if df is None or df.empty:
        raise RuntimeError("No rows returned. Try another symbol/period (e.g., SPY + 5y).")

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if "Close" not in df.columns:
        raise RuntimeError("Yahoo data missing 'Close' column.")

    return df


@st.cache_data(show_spinner=False)
def run_analysis(
    df: pd.DataFrame,
    w_short: int,
    w_long: int,
    lookback: int,
) -> tuple[pd.DataFrame, int]:
    """
    Compute realized volatility series and rolling percentile.

    Returns:
      (out_df, effective_lookback)
    """
    close = df["Close"].dropna()

    hv_s = compute_realized_vol(close, w_short)
    hv_l = compute_realized_vol(close, w_long)

    # Clamp lookback based on data availability (prevents "not enough datapoints" in common cases)
    effective_lookback = clamp_lookback_to_data(hv_s, int(lookback))

    pct = percentile_rank(hv_s, window=effective_lookback)

    out = pd.DataFrame({
        "Close": close,
        "HV_short": hv_s,
        "HV_long": hv_l,
        "HV_pct": pct,
    }).dropna()

    # Be slightly less strict: require enough rows for long window and percentile lookback,
    # but don't block unnecessarily if the dataset is otherwise usable.
    min_required = max(w_long, effective_lookback, 80)
    if len(out) < min_required:
        raise RuntimeError(
            "Not enough datapoints after rolling calculations. "
            "Increase period (e.g., 5y/max) or reduce windows/lookback."
        )

    return out, effective_lookback


# --- Plotting --------------------------------------------------------------

def plot_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(df.index, df["Close"])
    ax.set_title("Price (Adj)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    st.pyplot(fig, use_container_width=True)


def plot_vol(df: pd.DataFrame, w_short: int, w_long: int):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(df.index, df["HV_short"], label=f"HV {w_short}d")
    ax.plot(df.index, df["HV_long"], label=f"HV {w_long}d")

    p25 = df["HV_short"].quantile(0.25)
    p75 = df["HV_short"].quantile(0.75)
    mean = df["HV_short"].mean()

    ax.axhline(p75, linestyle="--", alpha=0.7, label="HV short 75%")
    ax.axhline(p25, linestyle="--", alpha=0.7, label="HV short 25%")
    ax.axhline(mean, linestyle="--", alpha=0.7, label="HV short mean")

    ax.set_title("Realized Volatility (annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vol")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate(rotation=45)
    st.pyplot(fig, use_container_width=True)


def plot_scatter_forward(df: pd.DataFrame, fwd_days: int):
    """
    HV(short) today vs average HV(short) in next fwd_days trading days.
    """
    if fwd_days < 5:
        st.info("Forward horizon too small for scatter; use >= 5.")
        return

    future_hv = df["HV_short"].rolling(window=fwd_days, min_periods=fwd_days).mean().shift(-fwd_days)
    tmp = pd.DataFrame({"hv_now": df["HV_short"], "hv_fwd": future_hv}).dropna()

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(tmp["hv_now"], tmp["hv_fwd"], alpha=0.6, s=18)

    x = tmp["hv_now"].values
    y = tmp["hv_fwd"].values
    if len(x) >= 10:
        b1, b0 = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        yline = b1 * xline + b0
        ax.plot(xline, yline, linewidth=2, label=f"fit: y={b1:.2f}x+{b0:.2f}")
        ax.legend(fontsize=8)

    ax.set_title(f"HV(short) vs Forward {fwd_days}d HV(short)")
    ax.set_xlabel("HV(short) now")
    ax.set_ylabel(f"Forward {fwd_days}d avg HV(short)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)


# --- Streamlit App ---------------------------------------------------------

def init_state():
    if "data" not in st.session_state:
        st.session_state.data = None
    if "analysis_df" not in st.session_state:
        st.session_state.analysis_df = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "last_fetch_params" not in st.session_state:
        st.session_state.last_fetch_params = None
    if "last_analyze_params" not in st.session_state:
        st.session_state.last_analyze_params = None
    if "boot_ran" not in st.session_state:
        st.session_state.boot_ran = False


def hard_fallback_params():
    # Recruiter-proof defaults that almost always work
    return {
        "symbol": "SPY",
        "period": "5y",
        "w_short": 20,
        "w_long": 60,
        "lookback": 126,
        "fwd_days": 20,
    }


def main():
    st.set_page_config(page_title="Volatility Dashboard (Yahoo Finance)", layout="wide")
    init_state()

    st.title("Volatility Dashboard (Yahoo Finance)")

    # Controls
    with st.container():
        c0, c1, c2, c3, c4, c5 = st.columns([1.0, 1.2, 1.2, 1.4, 1.2, 1.0])
        auto_run = c0.checkbox("Auto-run", value=True, help="Auto fetch + analyze on first load and when parameters change.")
        symbol = c1.text_input("Symbol", value="SPY").strip().upper()
        # IMPORTANT: default to 5y so charts show on first open
        period = c2.text_input("Period (e.g., 1y, 2y, 5y, max)", value="5y").strip().lower()
        w_short = c3.number_input("HV short window", min_value=5, max_value=400, value=20, step=1)
        w_long = c4.number_input("HV long window", min_value=5, max_value=800, value=60, step=1)
        fetch_clicked = c5.button("Fetch Data", use_container_width=True)

    # More settings row
    s1, s2, s3, s4 = st.columns([1.2, 1.2, 1.2, 2.4])
    # IMPORTANT: default lookback to 126 for higher chance of success on shorter periods too
    lookback = s1.number_input(
        "Percentile lookback (days)",
        min_value=30,
        max_value=2000,
        value=126,
        step=1,
        help="Rolling window used for HV(short) percentile rank.",
    )
    fwd_days = s2.number_input(
        "Forward horizon (days)",
        min_value=5,
        max_value=252,
        value=20,
        step=1,
        help="Used in the scatter: HV(short) now vs forward average HV(short).",
    )

    # Validate window logic
    if int(w_short) >= int(w_long):
        st.warning("Short window should be smaller than long window (e.g., 20 and 60).")

    # Buttons row (Analyze + utility)
    b1, b2, b3, b4 = st.columns([1, 1, 1, 3])
    analyze_clicked = b1.button("Analyze", use_container_width=True)
    clear_logs_clicked = b2.button("Clear log", use_container_width=True)
    clear_cache_clicked = b3.button("Clear cache", use_container_width=True)

    if clear_logs_clicked:
        st.session_state.logs = []

    if clear_cache_clicked:
        st.cache_data.clear()
        st.session_state.data = None
        st.session_state.analysis_df = None
        st.session_state.last_fetch_params = None
        st.session_state.last_analyze_params = None
        st.session_state.boot_ran = False
        log("Cache cleared (and state reset).")

    # --- Auto-run logic ----------------------------------------------------
    current_fetch_params = (symbol, period)
    current_analyze_params = (symbol, period, int(w_short), int(w_long), int(lookback), int(fwd_days))

    # On first load with auto_run, force a boot run so recruiter sees charts immediately
    should_boot = auto_run and not st.session_state.boot_ran

    should_autofetch = (
        auto_run
        and symbol
        and (should_boot or st.session_state.data is None or st.session_state.last_fetch_params != current_fetch_params)
    )

    # Auto-analyze after we have data for the current fetch params
    should_autoanalyze = (
        auto_run
        and st.session_state.data is not None
        and st.session_state.last_fetch_params == current_fetch_params
        and (should_boot or st.session_state.analysis_df is None or st.session_state.last_analyze_params != current_analyze_params)
        and (int(w_short) < int(w_long))
    )

    # Fetch action (manual or auto)
    if fetch_clicked or should_autofetch:
        if not symbol:
            st.error("Please enter a symbol (e.g., SPY).")
        else:
            try:
                log(f"Fetching Yahoo Finance data for {symbol} (period={period})...")
                with st.spinner(f"Fetching {symbol}..."):
                    df = fetch_yahoo(symbol, period)
                st.session_state.data = df
                st.session_state.analysis_df = None
                st.session_state.last_fetch_params = current_fetch_params
                st.session_state.last_analyze_params = None
                log(f"Received {len(df)} rows. Range: {df.index.min().date()} → {df.index.max().date()}")
            except Exception as e:
                st.session_state.data = None
                st.session_state.analysis_df = None
                st.session_state.last_fetch_params = None
                st.session_state.last_analyze_params = None
                st.session_state.boot_ran = True  # avoid infinite retry loop
                msg = (
                    f"Fetch failed: {e}\n\n"
                    "Tip: Yahoo/yfinance sometimes throttles (especially on Streamlit Cloud). "
                    "Try again, use a different ticker, or try period=5y/max."
                )
                log(f"Fetch failed: {e}")
                st.error(msg)

    # Analyze action (manual or auto) + fallback mechanism
    if analyze_clicked or should_autoanalyze:
        if st.session_state.data is None or st.session_state.data.empty:
            st.error("No data loaded. Click Fetch Data first.")
        else:
            if int(w_short) < 5 or int(w_long) < 5:
                st.error("HV windows should be at least 5.")
            elif int(w_short) >= int(w_long):
                st.error("Short window should be smaller than long window (e.g., 20 and 60).")
            else:
                try:
                    log(
                        f"Computing realized volatility (short={int(w_short)}, long={int(w_long)}), "
                        f"lookback={int(lookback)}..."
                    )
                    with st.spinner("Analyzing..."):
                        out, eff_lb = run_analysis(
                            st.session_state.data,
                            int(w_short),
                            int(w_long),
                            int(lookback),
                        )
                    st.session_state.analysis_df = out
                    st.session_state.last_analyze_params = current_analyze_params
                    st.session_state.boot_ran = True

                    if eff_lb != int(lookback):
                        log(f"Note: lookback clamped from {int(lookback)} → {eff_lb} (based on available data).")

                    hv_s_now = float(out["HV_short"].iloc[-1])
                    hv_l_now = float(out["HV_long"].iloc[-1])
                    pct_now = float(out["HV_pct"].iloc[-1])
                    log(
                        f"Done. Latest HV(short)={hv_s_now:.4f}, HV(long)={hv_l_now:.4f}, "
                        f"percentile={pct_now:.3f} (lookback={eff_lb})"
                    )
                except Exception as e:
                    # If auto-run: do a single hard fallback to ensure charts appear on first open
                    log(f"Analyze failed: {e}")
                    if auto_run and should_boot:
                        fb = hard_fallback_params()
                        log(
                            "Auto-run fallback activated to guarantee charts on first open: "
                            f"symbol={fb['symbol']}, period={fb['period']}, w_short={fb['w_short']}, "
                            f"w_long={fb['w_long']}, lookback={fb['lookback']}."
                        )
                        try:
                            with st.spinner("Applying safe defaults..."):
                                df = fetch_yahoo(fb["symbol"], fb["period"])
                                out, eff_lb = run_analysis(df, fb["w_short"], fb["w_long"], fb["lookback"])
                            st.session_state.data = df
                            st.session_state.analysis_df = out
                            st.session_state.last_fetch_params = (fb["symbol"], fb["period"])
                            st.session_state.last_analyze_params = (fb["symbol"], fb["period"], fb["w_short"], fb["w_long"], fb["lookback"], int(fwd_days))
                            st.session_state.boot_ran = True
                            log("Fallback successful. Charts should now be visible.")
                        except Exception as e2:
                            st.session_state.analysis_df = None
                            st.session_state.boot_ran = True
                            st.error(
                                f"Analyze failed: {e}\n\nFallback also failed: {e2}\n\n"
                                "Tip: Try again later (Yahoo throttling) or use a different ticker/period."
                            )
                    else:
                        st.session_state.analysis_df = None
                        st.session_state.boot_ran = True
                        st.error(
                            f"Analyze failed: {e}\n\n"
                            "Tip: Increase period (e.g., 5y/max), or reduce windows/lookback."
                        )

    # Snapshot
    st.subheader("Current Volatility Snapshot")
    snap = st.session_state.analysis_df

    cA, cB, cC, cD, cE = st.columns([1.1, 1.1, 1.1, 1.2, 1.5])
    if snap is not None and not snap.empty:
        hv_s_now = float(snap["HV_short"].iloc[-1])
        hv_l_now = float(snap["HV_long"].iloc[-1])
        pct_now = float(snap["HV_pct"].iloc[-1])
        regime, color = regime_from_percentile(pct_now)

        cA.metric("Current HV (short)", f"{hv_s_now*100:.2f}%")
        cB.metric("Current HV (long)", f"{hv_l_now*100:.2f}%")
        cC.metric("Percentile (short)", f"{pct_now:.1%}")
        cD.markdown(
            f"**Regime:** <span style='color:{color}; font-weight:700;'>{regime}</span>",
            unsafe_allow_html=True
        )

        # Download CSV
        csv_bytes = snap.to_csv(index=True).encode("utf-8")
        cE.download_button(
            label="Download analysis (CSV)",
            data=csv_bytes,
            file_name=f"{symbol}_volatility_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        cA.metric("Current HV (short)", "N/A")
        cB.metric("Current HV (long)", "N/A")
        cC.metric("Percentile (short)", "N/A")
        cD.markdown("**Regime:** N/A")

    # Results plots
    st.subheader("Results")
    if snap is not None and not snap.empty:
        p1, p2, p3 = st.columns(3)
        with p1:
            plot_price(snap)
        with p2:
            plot_vol(snap, int(w_short), int(w_long))
        with p3:
            plot_scatter_forward(snap, int(fwd_days))

        with st.expander("Show analysis table"):
            st.dataframe(snap.tail(300), use_container_width=True)
    else:
        st.info("Loading data... If this persists, try period=5y or click Fetch Data.")

    # Status log
    st.subheader("Status")
    st.text_area("Log", value="\n".join(st.session_state.logs), height=180)


if __name__ == "__main__":
    main()
