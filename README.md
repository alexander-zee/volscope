# VolScope

**VolScope** is an interactive **realized volatility analytics dashboard** built in Python.  
It is designed for exploratory analysis of volatility dynamics in financial markets, with a focus on **volatility regimes, persistence, and forward-looking behavior**.

The dashboard fetches market data, computes rolling volatility metrics, and presents them through clear, interactive visualizations.

---

## 🔧 Technologies

- Python  
- pandas, NumPy  
- matplotlib  
- Streamlit  
- Yahoo Finance API (yfinance)

---

## 📊 Features

Here’s what you can do with **VolScope**:

- **Market Data Ingestion**  
  Fetch daily adjusted price data for equities using Yahoo Finance.

- **Realized Volatility Calculation**  
  Compute annualized realized volatility from daily log returns using configurable rolling windows.

- **Multi-Horizon Analysis**  
  Compare short-term and long-term volatility dynamics side-by-side.

- **Volatility Regime Classification**  
  Classify current volatility levels using rolling percentile ranks (e.g. low, normal, high volatility).

- **Forward Volatility Analysis**  
  Analyze volatility persistence by comparing current volatility to future realized volatility averages.

- **Interactive Visualization**  
  Explore price levels, volatility time series, and cross-sectional scatter plots in an interactive dashboard.

---

## 🧠 The Process

I started by designing a clean **data pipeline** that separates data ingestion, transformation, and visualization.

First, daily closing prices are retrieved and cleaned. From these prices, log returns are computed and used to estimate **annualized realized volatility** over rolling windows.

Next, I implemented a **percentile-based framework** to contextualize current volatility relative to historical observations. This allows the dashboard to identify volatility regimes rather than relying on absolute thresholds.

To study volatility persistence, I added a forward-looking component that compares today’s volatility with the average realized volatility over the next 20 trading days, visualized through scatter plots and a fitted regression line.

Finally, I wrapped the analytics into a **Streamlit application**, focusing on clarity, responsiveness, and ease of experimentation.

---

## 📐 Methodology

Realized volatility is computed as:

HVₜ = std( log(Pₜ / Pₜ₋₁) ) × √252


Volatility regimes are determined using rolling percentile ranks of short-horizon volatility.  
Forward volatility is defined as the average realized volatility over the next 20 trading days.

This setup allows inspection of **volatility clustering and mean reversion behavior** commonly observed in financial markets.

---

## 📈 What I Learned

Through this project, I strengthened my understanding of:

- Time series volatility dynamics and clustering  
- Rolling-window statistics and percentile-based normalization  
- Designing reproducible financial data pipelines  
- Translating quantitative concepts into clear visual tools  
- Building interactive analytical dashboards for exploration rather than static reporting  

I also gained experience balancing **quantitative rigor** with **practical usability**.

---

## 🔄 Possible Improvements

Planned or potential extensions include:

- Integration of **implied volatility** from options data  
- Cross-asset volatility comparison  
- Volatility regime backtesting  
- Export functionality for downstream analysis  
- Additional statistical diagnostics for persistence and regime shifts  

---

## ▶️ Running the Project

To run VolScope locally:

pip install streamlit yfinance pandas numpy matplotlib
streamlit run streamlit_volatility_dashboard.py


The app will open in your browser at:  
`http://localhost:8501`

---

## 📌 Project Structure

- `streamlit_volatility_dashboard.py` — Streamlit application  
- `iv_dashboard.py` — Core analysis and experimentation  
- `README.md` — Project documentation  

---

## 📬 Contact

If you’re interested in quantitative finance, volatility modeling, or data-driven tooling, feel free to connect:

- **LinkedIn:** https://www.linkedin.com/in/alexanderzee/
