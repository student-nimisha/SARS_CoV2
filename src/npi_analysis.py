"""
npi_analysis.py

This script estimates the effect of NPIs and vaccination using regression
on individual reproduction numbers.

What it does:
- Uses Rc (individual reproduction number) as target
- Includes NPI values (with a lag), vaccination, age, and other factors
- Runs regression to estimate impact on transmission
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)

# NPI lag in days
NPI_LAG = 5

NPI_COLS = [
    "facial_coverings",
    "gathering_restr",
    "school_restr",
    "workplace_restr",
    "travel_restr",
    "stay_home",
]

SETTINGS = ["total","household","school","workplace","family","community"]

# Merge Rc data with NPI values
def build_regression_data(rc_df: pd.DataFrame, npi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Rc data with NPI values (lagged).
    """
    npi_lagged = npi_df.copy()
    npi_lagged["date"] = pd.to_datetime(npi_lagged["date"]) + pd.Timedelta(days=NPI_LAG)
    npi_lagged = npi_lagged.rename(columns={"date": "test_date_dt"})
    
    rc_df = rc_df.copy()
    rc_df["test_date_dt"] = pd.to_datetime(rc_df["test_date"])
    
    merged = rc_df.merge(npi_lagged, on="test_date_dt", how="left")
    
    # Fill missing values
    for c in NPI_COLS + ["school_holiday"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(merged[c].median())
    
    return merged

# Negative binomial log-likelihood
def nb_loglik(y: np.ndarray, mu: np.ndarray, k: float) -> float:
    """Compute NB log-likelihood."""
    from scipy.special import gammaln
    y = np.clip(y, 0, None)
    mu = np.clip(mu, 1e-6, None)
    k = max(k, 1e-4)
    ll = (gammaln(y + k) - gammaln(k) - gammaln(y + 1)
          + k * np.log(k / (k + mu))
          + y * np.log(mu / (k + mu)))
    return np.sum(ll)

# Negative Binomial regression using scipy
def neg_binomial_glm(y: np.ndarray, X: np.ndarray,
                     npi_indices: list = None) -> dict:
    """
    Fit Negative Binomial model:
    log(mu) = X * beta
    """
    from scipy.special import gammaln
    from scipy.optimize import minimize

    n, p = X.shape
    y = np.clip(y, 0, None)
    pen_idx = list(npi_indices) if npi_indices else []

    def nb_nll(params):
        beta = params[:p]
        log_k = params[p]
        k = np.exp(log_k)
        mu = np.clip(np.exp(X @ beta), 1e-8, 1e6)
        ll = (gammaln(y + k) - gammaln(k) - gammaln(y + 1)
              + k * np.log(k / (k + mu))
              + y * np.log(mu / (k + mu)))
        penalty = 0.05 * np.sum(beta[pen_idx] ** 2) if pen_idx else 0.0
        return -(np.sum(ll) - penalty)

    log_y = np.log(y + 0.5)
    try:
        beta0 = np.linalg.lstsq(X, log_y, rcond=None)[0]
    except Exception:
        beta0 = np.zeros(p)

    x0 = np.concatenate([beta0, [0.0]])

    res = minimize(nb_nll, x0, method="L-BFGS-B",
                   options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-7})
    
    beta_hat = res.x[:p]
    k_hat = np.exp(res.x[p])

    # Approximate standard errors
    eps = 1e-5
    H = np.zeros((p, p))
    for i in range(p):
        x_fwd = res.x.copy(); x_fwd[i] += eps
        x_bwd = res.x.copy(); x_bwd[i] -= eps
        g_fwd = _grad_nb_beta(x_fwd, X, y, p, pen_idx)
        g_bwd = _grad_nb_beta(x_bwd, X, y, p, pen_idx)
        H[i] = (g_fwd - g_bwd) / (2 * eps)

    try:
        Cov = np.linalg.inv(H + 1e-6 * np.eye(p))
        se = np.sqrt(np.abs(np.diag(Cov)))
    except Exception:
        se = np.ones(p) * 0.1

    from scipy import stats as _stats
    z = beta_hat / np.maximum(se, 1e-8)
    pvals = 2 * _stats.norm.sf(np.abs(z))

    return {"coef": beta_hat, "se": se, "pvalues": pvals, "k": k_hat}


def _grad_nb_beta(params, X, y, p, pen_idx):
    """Gradient for NB model."""
    beta = params[:p]
    log_k = params[p]
    k = np.exp(log_k)
    mu = np.clip(np.exp(X @ beta), 1e-8, 1e6)
    r = y - mu * (y + k) / (mu + k)
    g = X.T @ r
    for i in pen_idx:
        g[i] -= 2 * 0.05 * beta[i]
    return g


# Poisson regression (fallback)
def poisson_glm(y: np.ndarray, X: np.ndarray) -> dict:
    """
    Simple Poisson regression (used if needed).
    """
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(50):
        mu = np.exp(X @ beta)
        mu = np.clip(mu, 1e-6, 1e6)
        W  = np.diag(mu)
        z  = X @ beta + (y - mu) / mu

        try:
            XWX  = X.T @ W @ X + 1e-8 * np.eye(p)
            XWz  = X.T @ W @ z
            beta_new = np.linalg.solve(XWX, XWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    mu = np.exp(X @ beta)
    mu = np.clip(mu, 1e-6, 1e6)

    try:
        XWX = X.T @ np.diag(mu) @ X + 1e-8 * np.eye(p)
        Cov = np.linalg.inv(XWX)
        se  = np.sqrt(np.diag(Cov))
    except Exception:
        se = np.ones(p) * np.nan

    from scipy import stats as _stats
    z_stats = beta / (se + 1e-10)
    pvals   = 2 * _stats.norm.sf(np.abs(z_stats))

    return {"coef": beta, "se": se, "pvalues": pvals}


# Run regression for a given setting
def run_npi_regression(reg_data: pd.DataFrame, setting: str = "total") -> pd.DataFrame:
    """
    Run regression for a given setting (total, household, etc.).
    """
    y_col = f"rc_{setting}"
    if y_col not in reg_data.columns:
        return pd.DataFrame()
    
    data = reg_data.dropna(subset=[y_col]).copy()
    y    = np.clip(data[y_col].values.astype(float), 0, None)
    
    if len(y) < 50:
        return pd.DataFrame()
    
    feature_names = []
    X_parts = [np.ones((len(data), 1))]
    feature_names.append("intercept")
    
    for c in NPI_COLS:
        if c in data.columns:
            X_parts.append(data[c].values.reshape(-1, 1))
            feature_names.append(c)
    
    if "school_holiday" in data.columns:
        X_parts.append(data["school_holiday"].values.reshape(-1, 1))
        feature_names.append("school_holiday")
    
    if "vacc_status" in data.columns:
        X_parts.append((data["vacc_status"] == "one_dose").astype(float).values.reshape(-1, 1))
        X_parts.append((data["vacc_status"] == "two_doses").astype(float).values.reshape(-1, 1))
        feature_names += ["vacc_one_dose", "vacc_two_doses"]
    
    for ag in ["0-10","11-18","19-39","40-59"]:
        if "age_group" in data.columns:
            X_parts.append((data["age_group"] == ag).astype(float).values.reshape(-1, 1))
            feature_names.append(f"age_{ag}")
    
    if "variant" in data.columns:
        for v in ["Alpha","Eta","Delta","Omicron"]:
            X_parts.append((data["variant"] == v).astype(float).values.reshape(-1, 1))
            feature_names.append(f"variant_{v}")
    
    X = np.hstack(X_parts)
    
    npi_idx = [i for i, fn in enumerate(feature_names) if fn in NPI_COLS]
    fit = neg_binomial_glm(y, X, npi_indices=npi_idx)

    results = pd.DataFrame({
        "covariate":  feature_names,
        "beta":       fit["coef"],
        "se":         fit["se"],
        "pvalue":     fit["pvalues"],
        "ci_low":     fit["coef"] - 1.96 * fit["se"],
        "ci_high":    fit["coef"] + 1.96 * fit["se"],
        "setting":    setting,
    })
    
    return results


def run_all_settings(reg_data: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    for setting in SETTINGS:
        res = run_npi_regression(reg_data, setting)
        if not res.empty:
            all_results.append(res)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def vaccination_effect_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vaccination effect (IRR).
    """
    vacc_rows = results_df[results_df["covariate"].isin(["vacc_one_dose","vacc_two_doses"])].copy()
    vacc_rows["IRR"]       = np.exp(vacc_rows["beta"])
    vacc_rows["IRR_low"]   = np.exp(vacc_rows["ci_low"])
    vacc_rows["IRR_high"]  = np.exp(vacc_rows["ci_high"])
    vacc_rows["reduction_pct"] = (1 - vacc_rows["IRR"]) * 100
    return vacc_rows[["covariate","setting","IRR","IRR_low","IRR_high","reduction_pct"]]


def age_transmission_irr(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get age-wise transmission effect.
    """
    age_rows = results_df[results_df["covariate"].str.startswith("age_")].copy()
    age_rows["IRR"] = np.exp(age_rows["beta"])
    age_rows["age_group"] = age_rows["covariate"].str.replace("age_","")
    return age_rows[["age_group","setting","IRR","beta","ci_low","ci_high"]]