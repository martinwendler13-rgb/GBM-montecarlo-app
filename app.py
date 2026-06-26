# =============================================================================
# GARCH(1,1) + GBM Monte Carlo — Production Streamlit App
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
import yfinance as yf
from datetime import date
from scipy.optimize import minimize
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GBM Monte Carlo Simulation Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] { background: #f0f2f6; border-radius: 8px; padding: 0.6rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS = 252
GARCH_FLOOR  = 1e-12   # numerical floor for conditional variance
SIGMA_MAX    = 2.0     # annualised — above this warns of instability
MIN_OBS      = 60      # minimum price observations required


# =============================================================================
# SECTION 1 — DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def get_price_data(ticker: str, start: str) -> pd.Series | None:
    try:
        data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    except Exception:
        return None
    if data.empty:
        return None
    prices = data["Close"].dropna()
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    return prices


# =============================================================================
# SECTION 2 — GARCH(1,1)
# =============================================================================

@st.cache_data(show_spinner=False)
def estimate_garch(returns_tuple: tuple) -> dict:
    """
    MLE for GARCH(1,1). Input is a tuple so st.cache_data can hash it.
    All math is identical to the original notebook.
    """
    returns = np.asarray(returns_tuple, dtype=np.float64)
    v0 = float(np.var(returns))

    def _nll(params):
        omega, alpha, beta = params
        if (alpha + beta) >= 1.0:
            return 1e10
        T  = len(returns)
        v  = np.empty(T)
        v[0] = v0
        ll = 0.0
        for i in range(1, T):
            vi = omega + alpha * returns[i - 1] ** 2 + beta * v[i - 1]
            if vi <= 0.0:
                return 1e10
            v[i] = vi
            ll  += -np.log(vi) - returns[i] ** 2 / vi
        return -ll

    result = minimize(
        _nll,
        x0     = np.array([1e-6, 0.05, 0.9]),
        method = "L-BFGS-B",
        bounds = [(GARCH_FLOOR, None), (0.0, 1.0), (0.0, 1.0)],
    )

    omega, alpha, beta = result.x
    persistence        = alpha + beta
    long_run_var       = omega / (1.0 - persistence) if persistence < 1.0 else np.nan

    return {
        "omega":        omega,
        "alpha":        alpha,
        "beta":         beta,
        "persistence":  persistence,
        "long_run_var": long_run_var,
        "params":       result.x,
        "v0":           v0,
        "converged":    result.success,
    }


def compute_garch_variance(params: np.ndarray, returns: np.ndarray, v0: float) -> np.ndarray:
    """Conditional variance series with numerical floor for stability."""
    omega, alpha, beta = params
    T = len(returns)
    v = np.empty(T)
    v[0] = v0
    for i in range(1, T):
        v[i] = max(omega + alpha * returns[i - 1] ** 2 + beta * v[i - 1], GARCH_FLOOR)
    return v


# =============================================================================
# SECTION 3 — MONTE CARLO GBM
# =============================================================================

def simulate_gbm(
    S0:          float,
    mu:          float,
    sigma:       float,
    T_days:      int   = 126,
    n_sim:       int   = 10_000,
    dt:          float = 1.0 / TRADING_DAYS,
    random_seed: int   = 42,
) -> tuple[np.ndarray, dict]:
    """
    Vectorised GBM simulation. Geometry identical to original;
    inner loop replaced by a single matrix operation for speed.
    """
    rng = np.random.default_rng(random_seed)
    Z   = rng.standard_normal((n_sim, T_days))

    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths      = np.concatenate(
        [np.zeros((n_sim, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    paths = S0 * np.exp(log_paths)

    S_T      = paths[:, -1]
    pcts     = np.percentile(S_T, [1, 5, 50, 95, 99])
    sim_rets = S_T / S0 - 1.0

    var_95 = float(np.percentile(sim_rets, 5))
    var_99 = float(np.percentile(sim_rets, 1))
    es_95  = float(sim_rets[sim_rets <= var_95].mean())
    es_99  = float(sim_rets[sim_rets <= var_99].mean())

    summary = {
        "S_T": S_T,
        "percentiles": {
            "P1":  pcts[0], "P5":  pcts[1],
            "P50": pcts[2], "P95": pcts[3], "P99": pcts[4],
        },
        "prob_up":         float(np.mean(S_T > S0)),
        "expected_return": float(sim_rets.mean()),
        "VaR_95": var_95,
        "VaR_99": var_99,
        "ES_95":  es_95,
        "ES_99":  es_99,
    }

    return paths, summary


# =============================================================================
# SECTION 4 — RESISTANCE ANALYSIS
# =============================================================================

def analyse_resistance(paths: np.ndarray, S0: float, resistance: float) -> dict:
    """Computes all resistance-related statistics from the path matrix."""
    S_T      = paths[:, -1]
    hit_mask = paths >= resistance

    prob_break = float(np.mean(S_T > resistance))
    hit_any    = hit_mask.any(axis=1)
    prob_touch = float(np.mean(hit_any))

    # First-passage time: argmax returns 0 for rows that never hit — filter those
    first_hit_idx = np.argmax(hit_mask, axis=1)
    hitting_paths = first_hit_idx[hit_any]
    avg_hit_day   = float(hitting_paths.mean()) if len(hitting_paths) > 0 else None

    return {
        "prob_break":  prob_break,
        "prob_touch":  prob_touch,
        "avg_hit_day": avg_hit_day,
        "n_hitting":   int(hit_any.sum()),
    }


# =============================================================================
# SECTION 5 — STATISTICAL VALIDATION
# =============================================================================

def run_validation(log_returns: pd.Series, paths: np.ndarray) -> dict:
    lr          = log_returns.values
    sim_returns = np.log(paths[:, 1:] / paths[:, :-1]).ravel()

    jb_stat, jb_p = stats.jarque_bera(lr)

    sw_sample     = lr if len(lr) <= 5_000 else np.random.default_rng(0).choice(lr, 5_000, replace=False)
    sw_stat, sw_p = stats.shapiro(sw_sample)

    lb     = acorr_ljungbox(lr,      lags=[5, 10], return_df=True)
    lb_vol = acorr_ljungbox(lr ** 2, lags=[5, 10], return_df=True)

    # KS test: cap simulated sample for performance; fixed seed for reproducibility
    rng        = np.random.default_rng(0)
    sim_sample = sim_returns if len(sim_returns) < 5_000 else rng.choice(sim_returns, 5_000, replace=False)
    ks_stat, ks_p = stats.ks_2samp(lr, sim_sample)

    return {
        "jb":     {"stat": jb_stat, "p": jb_p},
        "sw":     {"stat": sw_stat, "p": sw_p},
        "lb":     lb,
        "lb_vol": lb_vol,
        "ks":     {"stat": ks_stat, "p": ks_p},
        "moments": {
            "emp_mean": lr.mean(),          "sim_mean": sim_returns.mean(),
            "emp_std":  lr.std(),           "sim_std":  sim_returns.std(),
            "emp_skew": stats.skew(lr),     "sim_skew": stats.skew(sim_returns),
            "emp_kurt": stats.kurtosis(lr), "sim_kurt": stats.kurtosis(sim_returns),
        },
    }


# =============================================================================
# SECTION 6 — VISUALISATIONS
# =============================================================================

def _apply_style(fig: plt.Figure, axes) -> None:
    """Uniform dark-friendly styling across all charts."""
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(color="#2a2a2a", linewidth=0.6)
    fig.patch.set_facecolor("#0e1117")


def plot_mc_paths(paths: np.ndarray, S0: float, ticker: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n_display = min(200, len(paths))
    for i in range(n_display):
        ax.plot(paths[i], color="#4c9be8", alpha=0.04, linewidth=0.7)
    ax.plot(np.median(paths, axis=0), color="#ffffff", linewidth=1.8, label="Median", zorder=5)
    ax.axhline(S0, color="#f0c040", linestyle="--", linewidth=1.2, label=f"S₀ = {S0:.2f}")
    ax.set_title(f"Monte Carlo Paths — {ticker}  ({n_display} shown)")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.legend(framealpha=0.2)
    _apply_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_fan_chart(
    paths: np.ndarray, S0: float, resistance: float, ticker: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t   = np.arange(paths.shape[1])
    med = np.median(paths, axis=0)
    p5  = np.percentile(paths,  5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    p1  = np.percentile(paths,  1, axis=0)
    p99 = np.percentile(paths, 99, axis=0)

    ax.fill_between(t, p1,  p99, color="#4c9be8", alpha=0.12, label="P1–P99 (98%)")
    ax.fill_between(t, p5,  p95, color="#4c9be8", alpha=0.28, label="P5–P95 (90%)")
    ax.plot(t, med,        color="#ffffff", linewidth=2,    label="Median")
    ax.axhline(S0,         color="#f0c040", linestyle="--", linewidth=1.2, label=f"S₀ = {S0:.2f}")
    ax.axhline(resistance, color="#e05252", linestyle="-",  linewidth=2.0, label=f"Resistance {resistance}")

    ax.set_title(f"Fan Chart vs Resistance — {ticker}")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price")
    ax.legend(framealpha=0.2)
    _apply_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_final_distribution(
    S_T: np.ndarray, S0: float, summary: dict, resistance: float
) -> plt.Figure:
    p   = summary["percentiles"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(S_T, bins=80, density=True, color="#4c9be8", alpha=0.65, label="Final prices")

    for val, lbl, col, ls, lw in [
        (p["P5"],    f"P5  {p['P5']:.1f}",      "#e05252", "--", 1.5),
        (p["P50"],   f"P50 {p['P50']:.1f}",     "#50c878", "--", 1.5),
        (p["P95"],   f"P95 {p['P95']:.1f}",     "#f0c040", "--", 1.5),
        (resistance, f"Resistance {resistance}", "#e05252", "-",  2.0),
        (S0,         f"S₀ {S0:.1f}",            "#ffffff", ":",  1.8),
    ]:
        ax.axvline(val, color=col, linestyle=ls, linewidth=lw, label=lbl)

    ax.set_title("Final Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.2, fontsize=8)
    _apply_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_returns_distribution(S_T: np.ndarray, S0: float, summary: dict) -> plt.Figure:
    rets = S_T / S0 - 1.0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(rets, bins=80, density=True, color="#9b6ec8", alpha=0.65)

    for val, lbl, col in [
        (rets.mean(),       "Mean return",                       "#ffffff"),
        (0.0,               "Break-even",                        "#f0c040"),
        (summary["VaR_95"], f"VaR 95% {summary['VaR_95']:.2%}", "#e05252"),
        (summary["VaR_99"], f"VaR 99% {summary['VaR_99']:.2%}", "#c03030"),
    ]:
        ax.axvline(val, color=col, linestyle="--", linewidth=1.4, label=lbl)

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Simulated Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.2, fontsize=8)
    _apply_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_conditional_volatility(
    conditional_variance: np.ndarray, prices: pd.Series, ticker: str
) -> plt.Figure:
    # Volatilidad anualizada
    ann_vol = np.sqrt(conditional_variance * TRADING_DAYS)

    fig, ax = plt.subplots(figsize=(12, 3.5))

    # Asegurar que X e Y tengan el mismo largo
    x_index = prices.index[-len(ann_vol):]

    ax.plot(
        x_index,
        ann_vol,
        color="#f0a030",
        linewidth=1.1,
        label="GARCH Conditional Volatility (annualised)"
    )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(f"GARCH(1,1) Conditional Volatility — {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised Volatility")
    ax.legend(framealpha=0.2)

    _apply_style(fig, ax)
    plt.tight_layout()
    return fig

def plot_qq(log_returns: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    stats.probplot(log_returns.values, dist="norm", plot=ax)
    ax.get_lines()[0].set(color="#4c9be8", markersize=2, alpha=0.6)
    ax.get_lines()[1].set(color="#e05252", linewidth=1.5)
    ax.set_title("Q-Q Plot — Log-Returns vs Normal")
    _apply_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_histogram_vs_normal(log_returns: pd.Series) -> plt.Figure:
    lr  = log_returns.values
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hist(lr, bins=60, density=True, color="#4c9be8", alpha=0.60, label="Empirical")
    x = np.linspace(lr.min(), lr.max(), 300)
    ax.plot(x, stats.norm.pdf(x, lr.mean(), lr.std()), "#e05252", lw=2, label="Normal PDF")
    ax.set_title("Log-Returns vs Normal")
    ax.set_xlabel("Log-return")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.2, fontsize=8)
    _apply_style(fig, ax)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 7 — VALIDATION DISPLAY
# =============================================================================

def _test_row(label: str, stat: float, p: float, reject_label: str, pass_label: str) -> None:
    """Renders a single statistical test with consistent formatting."""
    col_a, col_b, col_c = st.columns([2, 1.2, 1.2])
    col_a.markdown(f"**{label}**")
    col_b.markdown(f"stat = `{stat:.4f}`  \np = `{p:.5f}`")
    if p < 0.05:
        col_c.error(reject_label)
    else:
        col_c.success(pass_label)


def render_validation(validation: dict, garch: dict) -> None:
    jb   = validation["jb"]
    sw   = validation["sw"]
    lb   = validation["lb"]
    lb_v = validation["lb_vol"]
    ks   = validation["ks"]
    m    = validation["moments"]

    st.markdown("#### Normality")
    _test_row("Jarque-Bera",  jb["stat"], jb["p"], "Reject normality ❌", "Cannot reject normality ✅")
    _test_row("Shapiro-Wilk", sw["stat"], sw["p"], "Reject normality ❌", "Cannot reject normality ✅")

    st.markdown("#### Autocorrelation (Returns)")
    st.dataframe(lb.style.format({"lb_stat": "{:.3f}", "lb_pvalue": "{:.4f}"}), use_container_width=True)
    if (lb["lb_pvalue"] < 0.05).any():
        st.warning("Autocorrelation detected — GBM independence assumption may be violated.")
    else:
        st.success("No significant autocorrelation — GBM independence assumption holds. ✅")

    st.markdown("#### Volatility Clustering (Squared Returns)")
    st.dataframe(lb_v.style.format({"lb_stat": "{:.3f}", "lb_pvalue": "{:.4f}"}), use_container_width=True)
    if (lb_v["lb_pvalue"] < 0.05).any():
        st.success("Volatility clustering confirmed — GARCH estimation is justified. ✅")
    else:
        st.warning("No volatility clustering detected — GARCH motivation weaker for this sample.")

    st.markdown("#### KS Test — Empirical vs Simulated Returns")
    _test_row("Kolmogorov-Smirnov", ks["stat"], ks["p"],
              "Distributions differ significantly ⚠️",
              "Cannot reject equal distributions ✅")

    st.markdown("#### Moment Comparison")
    df_mom = pd.DataFrame({
        "Empirical": [m["emp_mean"], m["emp_std"],  m["emp_skew"], m["emp_kurt"]],
        "Simulated": [m["sim_mean"], m["sim_std"],  m["sim_skew"], m["sim_kurt"]],
    }, index=["Mean", "Std Dev", "Skewness", "Excess Kurtosis"])
    st.dataframe(df_mom.style.format("{:.6f}"), use_container_width=True)

    st.markdown("#### GARCH(1,1) Parameters")
    df_garch = pd.DataFrame({
        "Parameter": ["omega", "alpha", "beta", "alpha + beta", "Long-run variance"],
        "Value": [
            garch["omega"],
            garch["alpha"],
            garch["beta"],
            garch["alpha"] + garch["beta"],
            garch["long_run_var"],
        ],
    })
    st.dataframe(df_garch.style.format({"Value": "{:.6e}"}), use_container_width=True)


# =============================================================================
# SECTION 8 — INTERPRETATION
# =============================================================================

def build_interpretation(
    summary:   dict,
    res_stats: dict,
    val:       dict,
    sigma:     float,
) -> list[str]:
    lines      = []
    p          = summary["percentiles"]
    prob_break = res_stats["prob_break"]
    prob_touch = res_stats["prob_touch"]
    avg_day    = res_stats["avg_hit_day"]
    res_level  = res_stats["level"]

    # Resistance — final price
    if prob_break > 0.5:
        lines.append(
            f"**Resistance break (at expiry):** {prob_break*100:.1f}% probability of closing above "
            f"{res_level} at horizon end — model indicates a likely breakout. ✅"
        )
    elif prob_break > 0.15:
        lines.append(
            f"**Resistance break (at expiry):** {prob_break*100:.1f}% probability — possible but "
            f"not the dominant scenario. ⚠️"
        )
    else:
        lines.append(
            f"**Resistance break (at expiry):** Only {prob_break*100:.1f}% probability — breakout "
            f"is unlikely under current model parameters. ❌"
        )

    # Resistance — path-wise touch
    if prob_touch > 0.5:
        lines.append(
            f"**Resistance touch (intraperiod):** {prob_touch*100:.1f}% of paths reach {res_level} "
            f"at least once — the level is regularly tested during the horizon."
        )
    elif prob_touch > 0.1:
        lines.append(
            f"**Resistance touch (intraperiod):** {prob_touch*100:.1f}% of paths touch {res_level} "
            f"at some point — infrequent contact with the resistance level."
        )
    else:
        lines.append(
            f"**Resistance touch (intraperiod):** Only {prob_touch*100:.1f}% of paths reach "
            f"{res_level} at any point — the level is unlikely to be tested."
        )

    # First-passage time
    if avg_day is not None:
        lines.append(
            f"**First-passage time:** Among paths that reach the resistance, average time to "
            f"first contact is **{avg_day:.1f} trading days** (~{avg_day / 21:.1f} months)."
        )

    # Upper tail vs resistance
    if p["P95"] > res_level:
        lines.append(
            f"**Tail coverage:** P95 = {p['P95']:.2f} exceeds resistance {res_level} — "
            f"the optimistic scenario includes a breakout."
        )
    else:
        lines.append(
            f"**Tail coverage:** Even the P95 scenario ({p['P95']:.2f}) falls below resistance "
            f"{res_level} — the level represents a meaningful ceiling."
        )

    # Directional bias
    if summary["prob_up"] > 0.55:
        lines.append(
            f"**Directional bias: Bullish** — {summary['prob_up']*100:.1f}% of paths end above "
            f"today's price. Expected return over the horizon: {summary['expected_return']*100:.2f}%."
        )
    elif summary["prob_up"] < 0.45:
        lines.append(
            f"**Directional bias: Bearish** — only {summary['prob_up']*100:.1f}% of paths end "
            f"above today's price. Expected return: {summary['expected_return']*100:.2f}%."
        )
    else:
        lines.append(
            f"**Directional bias: Neutral** — {summary['prob_up']*100:.1f}% of paths end above "
            f"today's price. Expected return: {summary['expected_return']*100:.2f}%."
        )

    # Risk metrics
    lines.append(
        f"**Risk (VaR / ES):** "
        f"95% VaR = {summary['VaR_95']*100:.2f}%  |  "
        f"95% ES = {summary['ES_95']*100:.2f}%  |  "
        f"99% VaR = {summary['VaR_99']*100:.2f}%  |  "
        f"99% ES = {summary['ES_99']*100:.2f}%  *(horizon returns)*"
    )

    # Model diagnostics
    jb_rejected = val["jb"]["p"] < 0.05
    autocorr    = (val["lb"]["lb_pvalue"] < 0.05).any()
    clustering  = (val["lb_vol"]["lb_pvalue"] < 0.05).any()

    if jb_rejected:
        lines.append(
            "**Normality:** Log-returns exhibit fat tails / skewness (Jarque-Bera rejected). "
            "This is standard in equity data — GBM is a first-order approximation."
        )
    else:
        lines.append(
            "**Normality:** Log-returns cannot reject normality at 5% — "
            "GBM distributional assumption holds for this sample."
        )

    if not autocorr:
        lines.append(
            "**Serial independence:** No significant autocorrelation in returns — "
            "the GBM i.i.d. increment assumption is supported. ✅"
        )
    else:
        lines.append(
            "**Serial independence:** Autocorrelation detected in returns — "
            "GBM independence assumption is violated; interpret simulation with caution. ⚠️"
        )

    if clustering:
        lines.append(
            "**Volatility clustering:** Squared returns are autocorrelated — "
            "time-varying volatility is present, directly motivating the GARCH(1,1) filter. ✅"
        )
    else:
        lines.append(
            "**Volatility clustering:** No significant clustering in squared returns for this "
            "sample — GARCH motivation is weaker, though the model remains valid. ⚠️"
        )

    if sigma > SIGMA_MAX:
        lines.append(
            f"**Volatility warning:** Estimated sigma = {sigma:.2%} (annualised) is unusually "
            "high. Verify that the input data and date range are correct."
        )

    return lines


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    st.title("📈 GBM Monte Carlo Simulation Model")
    st.caption(
        "Volatility-adjusted Monte Carlo simulation · "
        "GARCH conditional variance · Full statistical validation"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Model Parameters")

    ticker      = st.sidebar.text_input("Ticker", value="SAP").upper().strip()
    start_date  = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
    n_sim       = int(st.sidebar.number_input(
                    "Simulations", min_value=100, max_value=100_000,
                    value=10_000, step=1_000))
    T_days      = int(st.sidebar.number_input(
                    "Horizon (trading days)", min_value=10,
                    max_value=504, value=126, step=1))
    resistance  = float(st.sidebar.number_input(
                    "Resistance Level", value=212.0, step=0.5))
    random_seed = int(st.sidebar.number_input("Random Seed", value=42, step=1))

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Horizon ≈ {T_days / 21:.1f} months · {T_days / 252:.2f} years")

    if n_sim * T_days > 5_000_000:
        st.warning(
            f"Large simulation: {n_sim:,} × {T_days} = {n_sim * T_days:,} steps — may be slow."
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    with st.spinner(f"Downloading {ticker}..."):
        prices = get_price_data(ticker, str(start_date))

    if prices is None or len(prices) < MIN_OBS:
        st.error(
            f"Could not retrieve sufficient data for **{ticker}** from {start_date}. "
            f"Minimum required: {MIN_OBS} observations. "
            "Check the ticker symbol or choose an earlier start date."
        )
        st.stop()

    log_returns = np.log(prices / prices.shift(1)).dropna()

    st.sidebar.success(
        f"**{len(prices)}** observations  \n"
        f"{prices.index[0].date()} → {prices.index[-1].date()}"
    )

    # ── GARCH ─────────────────────────────────────────────────────────────────
    with st.spinner("Estimating GARCH(1,1)..."):
        garch = estimate_garch(tuple(log_returns.values))

    if not garch["converged"]:
        st.warning(
            "GARCH optimisation did not fully converge. "
            "Parameters may be unreliable — consider a longer data history."
        )
    if garch["persistence"] >= 0.999:
        st.warning(
            f"GARCH persistence (α + β = {garch['persistence']:.4f}) is near unit-root. "
            "Variance forecasts may be unreliable."
        )

    returns_arr          = log_returns.values
    conditional_variance = compute_garch_variance(garch["params"], returns_arr, garch["v0"])

    # Sigma: average over last 5 observations to dampen end-point noise
    sigma = float(np.sqrt(np.mean(conditional_variance[-5:]) * TRADING_DAYS))
    mu    = float(np.mean(returns_arr) * TRADING_DAYS + 0.5 * sigma ** 2)
    S0    = float(prices.iloc[-1])

    if sigma > SIGMA_MAX:
        st.warning(f"Annualised sigma = {sigma:.2%} is very high. Please verify the input data.")

    # ── Simulation ────────────────────────────────────────────────────────────
    with st.spinner("Running Monte Carlo..."):
        paths, summary = simulate_gbm(
            S0=S0, mu=mu, sigma=sigma,
            T_days=T_days, n_sim=n_sim, random_seed=random_seed,
        )

    res_stats          = analyse_resistance(paths, S0, resistance)
    res_stats["level"] = resistance

    validation = run_validation(log_returns, paths)

    S_T = summary["S_T"]
    p   = summary["percentiles"]

    # =========================================================================
    # A — KEY METRICS
    # =========================================================================
    st.header("Key Metrics")

    r1 = st.columns(4)
    r1[0].metric("Current Price (S₀)", f"{S0:.2f}")
    r1[1].metric("Sigma — annualised",  f"{sigma:.2%}")
    r1[2].metric("Mu — annualised",     f"{mu:.2%}")
    r1[3].metric("Expected Return",     f"{summary['expected_return']*100:.2f}%")

    r2 = st.columns(4)
    r2[0].metric("P(break resistance)", f"{res_stats['prob_break']*100:.1f}%")
    r2[1].metric("P(touch resistance)", f"{res_stats['prob_touch']*100:.1f}%")
    avg_day = res_stats["avg_hit_day"]
    avg_str = f"{avg_day:.1f} days" if avg_day is not None else "—"
    r2[2].metric("Avg First-Hit Day",   avg_str)
    r2[3].metric("P50 / P95",           f"{p['P50']:.1f} / {p['P95']:.1f}")

    r3 = st.columns(4)
    r3[0].metric("VaR 95%", f"{summary['VaR_95']*100:.2f}%")
    r3[1].metric("ES  95%", f"{summary['ES_95']*100:.2f}%")
    r3[2].metric("VaR 99%", f"{summary['VaR_99']*100:.2f}%")
    r3[3].metric("ES  99%", f"{summary['ES_99']*100:.2f}%")

    st.markdown("---")

    # =========================================================================
    # B — CHARTS
    # =========================================================================
    st.header("Simulation")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monte Carlo Paths")
        st.pyplot(plot_mc_paths(paths, S0, ticker))
    with col2:
        st.subheader("Fan Chart vs Resistance")
        st.pyplot(plot_fan_chart(paths, S0, resistance, ticker))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Final Price Distribution")
        st.pyplot(plot_final_distribution(S_T, S0, summary, resistance))
    with col4:
        st.subheader("Return Distribution + VaR")
        st.pyplot(plot_returns_distribution(S_T, S0, summary))

    st.subheader("GARCH Conditional Volatility")
    st.pyplot(plot_conditional_volatility(conditional_variance, prices, ticker))

    st.markdown("---")

    st.header("Return Diagnostics")
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Q-Q Plot")
        st.pyplot(plot_qq(log_returns))
    with col6:
        st.subheader("Log-Returns vs Normal")
        st.pyplot(plot_histogram_vs_normal(log_returns))

    st.markdown("---")

    # =========================================================================
    # C — VALIDATION
    # =========================================================================
    with st.expander("Statistical Validation", expanded=False):
        render_validation(validation, garch)

    # =========================================================================
    # D — INTERPRETATION
    # =========================================================================
    with st.expander("Model Interpretation", expanded=True):
        lines = build_interpretation(summary, res_stats, validation, sigma)
        for line in lines:
            st.markdown(f"- {line}")


main()
