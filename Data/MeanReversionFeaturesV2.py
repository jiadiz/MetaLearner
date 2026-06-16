
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

from sklearn.linear_model import LinearRegression

def residual_momentum(series, length, skip=0):
    """
    Sum of residuals over 'length' days, optionally skipping the most recent 'skip' days.
   (series is a 1D array-like with possible NaNs)
    """
    series = np.asarray(series)
    series = series[~np.isnan(series)]
    if len(series) < length + skip:
        return np.nan
    if skip > 0:
        window_vals = series[-(length + skip):-skip]
    else:
        window_vals = series[-length:]
    return window_vals.sum()

def residual_reversal(series, length):
    """
    Short-term reversal: negative sum of last 'length' residuals.
    """
    series = np.asarray(series)
    series = series[~np.isnan(series)]
    if len(series) < length:
        return np.nan
    window_vals = series[-length:]
    return -window_vals.sum()

def residual_rsi(series, length):
    """
    RSI-style indicator on residuals.
    """
    series = np.asarray(series)
    series = series[~np.isnan(series)]
    if len(series) < length + 1:
        return np.nan
    window_vals = series[-(length + 1):]
    diffs = np.diff(window_vals)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0  # all gains
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def residual_streak_length(series):
    """
    Count how many consecutive days (backward) residual > 0.
    """
    series = np.asarray(series)
    series = series[~np.isnan(series)]
    if len(series) == 0:
        return np.nan
    streak = 0
    for v in series[::-1]:
        if v > 0:
            streak += 1
        else:
            break
    return streak

def update_ema(prev_ema, new_value, alpha):
    """
    One-step EMA update.
    """
    if np.isnan(new_value):
        return prev_ema
    if np.isnan(prev_ema):
        return new_value
    return alpha * new_value + (1.0 - alpha) * prev_ema


def _simple_ols(x: np.ndarray, y: np.ndarray):
    """
    Closed-form OLS for y = intercept + beta * x, where x and y are 1-D
    numpy arrays of the same length. Returns (intercept, beta).

    If x has zero variance (degenerate window), returns (nan, nan).
    """
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    denom = float((x_centered * x_centered).sum())
    if denom == 0.0 or not np.isfinite(denom):
        return np.nan, np.nan
    beta = float((x_centered * (y - y_mean)).sum() / denom)
    intercept = float(y_mean - beta * x_mean)
    return intercept, beta


def _multivariate_ols(x: np.ndarray, y: np.ndarray):
    """
    OLS with intercept for y = intercept + X @ beta.

    Parameters
    ----------
    x : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)

    Returns
    -------
    intercept : float
    betas : np.ndarray, shape (p,)
    """
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[0] < x.shape[1] + 1:
        nan = np.nan
        return nan, np.full(x.shape[1] if x.ndim == 2 else 0, nan)
    x_aug = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(x_aug, y, rcond=None)
        if rank < x_aug.shape[1]:
            nan = np.nan
            return nan, np.full(x.shape[1], nan)
        intercept = float(coeffs[0])
        betas = coeffs[1:].astype(float)
        return intercept, betas
    except Exception:
        nan = np.nan
        return nan, np.full(x.shape[1], nan)


def _fit_ols(x_train: np.ndarray, y_train: np.ndarray):
    """
    Fit OLS on a training window. Accepts univariate (1-D x) or multivariate
    (2-D x with shape n x p). Returns (intercept, beta_or_betas).
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if x_train.ndim == 1:
        intercept, beta = _simple_ols(x_train, y_train)
        return intercept, beta
    return _multivariate_ols(x_train, y_train)


def _predict_ols(intercept: float, beta, x_today) -> float:
    x_today = np.asarray(x_today, dtype=float)
    if np.isnan(intercept):
        return np.nan
    if np.isscalar(beta) or (isinstance(beta, np.ndarray) and beta.ndim == 0):
        return float(intercept + float(beta) * float(x_today))
    x_today = x_today.ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    if len(x_today) != len(beta) or np.any(np.isnan(beta)):
        return np.nan
    return float(intercept + float(np.dot(x_today, beta)))


def _align_target_and_x(
    target_series: pd.Series,
    X: pd.Series | pd.DataFrame,
) -> tuple[pd.Series, pd.Series | pd.DataFrame]:
    """
    Align target (y) and predictors (X) to dates where all values are finite.
    Caller is responsible for passing price or return series consistently.
    """
    target_series = target_series.copy()
    if isinstance(X, pd.Series):
        combined = pd.concat(
            [target_series.rename("_y"), X.rename("_x")],
            axis=1,
        ).dropna()
        return combined["_y"], combined["_x"]
    combined = pd.concat([target_series.rename("_y"), X], axis=1).dropna()
    return combined["_y"], combined.drop(columns="_y")


def fit_ar1(y):
    """
    Fit AR(1): y_t = alpha + beta * y_{t-1} + eps.
    Returns (alpha, beta).
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    y_arr = y_arr[~np.isnan(y_arr)]
    if len(y_arr) < 3:
        return np.nan, np.nan
    return _simple_ols(y_arr[:-1], y_arr[1:])


def forecast_k_steps_ar1(alpha: float,
                         beta: float,
                         y_t: float,
                         k: int = 21):
    """
    AR(1) forecast helpers computed in a single pass. Returns a 2-tuple:

        (y_forecast, y_forecast_sum)
    """
    if abs(1.0 - beta) < 1e-12:
        y_forecast = y_t + k * alpha
        y_forecast_sum = k * y_t + alpha * k * (k + 1) / 2.0
        return float(y_forecast), float(y_forecast_sum)

    mu = alpha / (1.0 - beta)
    beta_k = beta ** k
    y_forecast = mu + beta_k * (y_t - mu)
    geom = beta * (1.0 - beta_k) / (1.0 - beta)
    y_forecast_sum = k * mu + (y_t - mu) * geom
    return float(y_forecast), float(y_forecast_sum)


def create_residual_mean_reversion_features(
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    forecast_horizon: int = 21,
):
    """
    OLS of y on x using all but the last observation (the training window),
    then derive residual-based diagnostics.

    Parameters
    ----------
    y_arr : np.ndarray, shape (n,)
        Target values; last element is "today".
    x_arr : np.ndarray, shape (n,) or (n, p)
        Predictor(s); univariate (n,) or multivariate (n, p). Last row is today.
    forecast_horizon : int, default 21
        Holding horizon for AR(1) residual forecasts.

    Returns
    -------
    9-tuple of floats:
        (y_today, pair_beta, adf_pval, z_score_today, resid_std,
         resid_today, resid_forecast, resid_forecasted_change,
         resid_forecast_sum)

    For multivariate x, ``pair_beta`` is the first predictor's coefficient
    (backward-compatible scalar); all coefficients are also available via the
    cache key ``betas`` when fill_missing stores multivariate fits.
    """
    y_arr = np.asarray(y_arr, dtype=float).ravel()
    x_arr = np.asarray(x_arr, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.ravel()
        mask = ~(np.isnan(y_arr) | np.isnan(x_arr))
        y_arr = y_arr[mask]
        x_arr = x_arr[mask]
        multivariate = False
    else:
        mask = ~np.isnan(y_arr)
        for j in range(x_arr.shape[1]):
            mask &= ~np.isnan(x_arr[:, j])
        y_arr = y_arr[mask]
        x_arr = x_arr[mask]
        multivariate = True

    if len(y_arr) < 3:
        nan = np.nan
        return nan, nan, nan, nan, nan, nan, nan, nan, nan

    x_train = x_arr[:-1] if multivariate else x_arr[:-1]
    y_train = y_arr[:-1]
    x_today = x_arr[-1]

    intercept, beta_or_betas = _fit_ols(x_train, y_train)
    if multivariate:
        betas = np.asarray(beta_or_betas, dtype=float)
        pair_beta = float(betas[0]) if len(betas) else np.nan
        if np.any(np.isnan(betas)):
            pair_beta = np.nan
    else:
        pair_beta = beta_or_betas
        if np.isnan(pair_beta):
            nan = np.nan
            return float(y_arr[-1]), nan, nan, nan, nan, nan, nan, nan, nan

    y_pred_train = np.array([
        _predict_ols(intercept, beta_or_betas, x_train[i])
        for i in range(len(y_train))
    ])
    if np.any(np.isnan(y_pred_train)):
        nan = np.nan
        y_today = float(y_arr[-1])
        return y_today, nan, nan, nan, nan, nan, nan, nan, nan

    residuals = y_train - y_pred_train
    std = float(residuals.std(ddof=1))
    y_today = float(y_arr[-1])
    resid_today = float(y_today - _predict_ols(intercept, beta_or_betas, x_today))
    z_score_today = resid_today / std if std != 0.0 else np.nan

    try:
        adf_pval = float(adfuller(residuals, autolag='AIC')[1])
    except Exception:
        adf_pval = np.nan

    ar1_alpha, ar1_beta = fit_ar1(residuals)
    resid_forecast, resid_forecast_sum = forecast_k_steps_ar1(
        ar1_alpha, ar1_beta, resid_today, k=forecast_horizon + 1)
    resid_forecasted_change = resid_forecast - resid_today

    return (y_today, float(pair_beta), adf_pval, z_score_today, std,
            resid_today, resid_forecast, resid_forecasted_change,
            resid_forecast_sum)


def fill_missing_mean_reversion_features(
    available_mean_reversion_features_per_ticker: dict,
    target_series: pd.Series,
    X: pd.Series | pd.DataFrame,
    ticker: str,
    series_type: str,
    lookback: int,
    sector_type: str,
    verbose: bool = False,
):
    """
    Fill one mean-reversion cache bucket for a single (target, predictor) pair.

    Parameters
    ----------
    available_mean_reversion_features_per_ticker : dict
        Hierarchical cache mutated in place.
    target_series : pd.Series
        The y variable (price or return, depending on ``series_type``).
    X : pd.Series or pd.DataFrame
        Predictor(s). A Series for one-factor models (sector ETF, SP500); a
        DataFrame for multi-factor models (e.g. Fama-French). Must already be
        in the same units as ``target_series`` (returns if series_type='return').
    ticker : str
        Stock ticker label for the cache hierarchy.
    series_type : {'price', 'return'}
        Used only for naming: ``mean_reversion_{sector_type}_{series_type}_d{lookback}``.
        Caller must pass return series when series_type='return'.
    lookback : int
        Rolling regression window length.
    sector_type : str
        Naming label for the predictor set (e.g. 'sector', 'econ', 'fama').
    verbose : bool
        Print progress messages.
    """
    if series_type not in ('price', 'return'):
        raise ValueError(f"series_type must be 'price' or 'return', got {series_type!r}")

    def identify_ticker_feature_location(cache, tkr, mean_reversion_type):
        if tkr not in cache:
            cache[tkr] = {}
            if verbose:
                print(f'{tkr} data created')
        if mean_reversion_type not in cache[tkr]:
            cache[tkr][mean_reversion_type] = {}
            if verbose:
                print(f'{tkr} {mean_reversion_type} data created')
        return cache[tkr][mean_reversion_type]

    y, x = _align_target_and_x(target_series, X)
    if y.empty:
        return

    y_arr = y.values
    if isinstance(x, pd.Series):
        x_arr = x.values
        x_columns = None
    else:
        x_arr = x.values
        x_columns = list(x.columns)

    mean_reversion_type = f'mean_reversion_{sector_type}_{series_type}_d{lookback}'
    data = identify_ticker_feature_location(
        available_mean_reversion_features_per_ticker, ticker, mean_reversion_type)

    for pos, date in enumerate(y.index):
        existing = data.get(date)
        needs_backfill = (
            isinstance(existing, dict)
            and 'y_today' in existing
            and ('resid_forecast' not in existing or 'resid_forecast_sum' not in existing)
        )
        if (date not in data) or needs_backfill:
            start = max(0, pos - lookback)
            y_win = y_arr[start: pos + 1]
            x_win = x_arr[start: pos + 1]
            if len(y_win) < lookback + 1:
                continue
            (y_today, beta, adf_p, z_score_today, resid_std,
             resid_today, resid_forecast, resid_forecasted_change,
             resid_forecast_sum) = create_residual_mean_reversion_features(y_win, x_win)

            entry = {
                'y_today': y_today,
                'beta': beta,
                'adf_p': adf_p,
                'z_score_today': z_score_today,
                'resid_std': resid_std,
                'resid_today': resid_today,
                'resid_forecast': resid_forecast,
                'resid_forecasted_change': resid_forecasted_change,
                'resid_forecast_sum': resid_forecast_sum,
            }
            if x_columns is not None and len(x_columns) > 1:
                _, betas = _fit_ols(x_win[:-1], y_win[:-1])
                entry['betas'] = {
                    col: float(betas[i]) if i < len(betas) else np.nan
                    for i, col in enumerate(x_columns)
                }
            data[date] = entry

    if verbose and len(y.index):
        print('Data until ', y.index[-1], f'{mean_reversion_type} data created')


def create_mean_reversion_variants(
    available_mean_reversion_features_per_ticker: dict,
    ticker: str,
    windows: list,
    sector_types: list[str],
    is_price_series: bool = True,
):
    """
    Expand cached mean-reversion base features into the full variant column set.

    Parameters
    ----------
    sector_types : list[str]
        Predictor labels to expand (e.g. ``['sector', 'econ']`` or
        ``['sector', 'econ', 'fama']``). Must match the ``sector_type`` values
        passed to ``fill_missing_mean_reversion_features`` / the keys stored in
        the cache.
    """
    return_or_price = 'price' if is_price_series else 'return'
    df = pd.DataFrame()
    ticker_cache = available_mean_reversion_features_per_ticker[ticker]

    for sector_type in sector_types:
        for window in windows:
            cache_key = f'mean_reversion_{sector_type}_{return_or_price}_d{window}'
            if cache_key not in ticker_cache:
                continue
            bucket = ticker_cache[cache_key]

            ys = pd.Series(bucket).apply(lambda x: x['y_today'] if 'y_today' in x else np.nan)
            beta = pd.Series(bucket).apply(lambda x: x['beta'] if 'beta' in x else np.nan)
            z = pd.Series(bucket).apply(lambda x: x['z_score_today'] if 'z_score_today' in x else np.nan)
            adf_p = pd.Series(bucket).apply(lambda x: x['adf_p'] if 'adf_p' in x else np.nan)
            std = pd.Series(bucket).apply(lambda x: x['resid_std'] if 'resid_std' in x else np.nan)
            resid = pd.Series(bucket).apply(lambda x: x['resid_today'] if 'resid_today' in x else np.nan)
            resid_forecast = pd.Series(bucket).apply(
                lambda x: x['resid_forecast'] if isinstance(x, dict) and 'resid_forecast' in x else np.nan)
            resid_forecast_sum = pd.Series(bucket).apply(
                lambda x: x['resid_forecast_sum'] if isinstance(x, dict) and 'resid_forecast_sum' in x else np.nan)

            if window >= 252:
                MOM_12_LEN = 252
            else:
                MOM_06_LEN = 126 - 21

            MOM_12_SKIP = 21
            MOM_3M_LEN = 63
            REV_5D_LEN = 5
            RSI_LEN = 14
            MACD_FAST = 12
            MACD_SLOW = 26
            MACD_SIGNAL = 9

            if window >= 252:
                res_mom_12_1 = resid.expanding().apply(
                    lambda x: residual_momentum(x, length=MOM_12_LEN, skip=MOM_12_SKIP))
                res_mom_long_1 = res_mom_12_1
            else:
                res_mom_06_1 = resid.expanding().apply(
                    lambda x: residual_momentum(x, length=MOM_06_LEN, skip=MOM_12_SKIP))
                res_mom_long_1 = res_mom_06_1

            res_rsi_14 = resid.expanding().apply(lambda x: residual_rsi(x, RSI_LEN), raw=False)
            res_mom_3m = resid.expanding(MOM_3M_LEN + 1).apply(
                lambda x: residual_momentum(x, length=MOM_3M_LEN, skip=0))
            res_rev_5d = resid.expanding().apply(lambda x: residual_reversal(x, length=REV_5D_LEN))
            res_streak = resid.expanding().apply(lambda x: residual_streak_length(x))

            ema_fast = resid.ewm(span=MACD_FAST, adjust=False).mean()
            ema_slow = resid.ewm(span=MACD_SLOW, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
            macd_hist = macd_line - macd_signal

            r_or_p = 'p' if is_price_series else 'r'
            prefix = sector_type

            if is_price_series:
                stock_price = ys
                D = window
                df[f'{prefix}_{r_or_p}_beta_d{window}'] = beta
                df[f'{prefix}_{r_or_p}_resid_std_d{window }'] = std / stock_price
                df[f'{prefix}_{r_or_p}_resid_z_d{window }'] = z
                df[f'{prefix}_{r_or_p}_resid_total_deviation_per_price_d{window}'] = (std * z) / stock_price
                df[f'{prefix}_{r_or_p}_resid_adf_p_d{window }'] = adf_p
                df[f'{prefix}_{r_or_p}_z_times_1_minus_p_value_{window }'] = z * (1 - adf_p)
                df[f'{prefix}_{r_or_p}_resid_std_times_z_times_1_minus_p_value_{window }'] = (
                    (std * z * (1 - adf_p)) / stock_price)
                if D >= 252:
                    df[f'{prefix}_{r_or_p}_res_mom_12_1_{window }'] = res_mom_long_1 / stock_price
                else:
                    df[f'{prefix}_{r_or_p}_res_mom_06_1_{window }'] = res_mom_long_1 / stock_price
                df[f'{prefix}_{r_or_p}_res_rev_5d_{window }'] = res_rev_5d / stock_price
                df[f'{prefix}_{r_or_p}_res_rsi_14_{window }'] = (res_rsi_14 * std * z * adf_p) / stock_price
                df[f'{prefix}_{r_or_p}_res_streak_{window }'] = (res_streak * std * z * adf_p) / stock_price
                df[f'{prefix}_{r_or_p}_macd_line_{window }'] = macd_line / stock_price
                df[f'{prefix}_{r_or_p}_macd_signal_{window }'] = macd_signal / stock_price
                df[f'{prefix}_{r_or_p}_macd_hist_{window }'] = macd_hist / stock_price
                df[f'{prefix}_{r_or_p}_resid_forecast_{window}'] = resid_forecast / stock_price
                df[f'{prefix}_{r_or_p}_resid_forecast_sum_{window}'] = resid_forecast_sum / stock_price
            else:
                D = window
                df[f'{prefix}_{r_or_p}_beta_d{window}'] = beta
                df[f'{prefix}_{r_or_p}_resid_std_d{window }'] = std
                df[f'{prefix}_{r_or_p}_resid_z_d{window }'] = z
                df[f'{prefix}_{r_or_p}_resid_total_deviation_per_price_d{window}'] = (std * z)
                df[f'{prefix}_{r_or_p}_resid_adf_p_d{window }'] = adf_p
                df[f'{prefix}_{r_or_p}_z_times_1_minus_p_value_{window }'] = z * (1 - adf_p)
                df[f'{prefix}_{r_or_p}_resid_std_times_z_times_1_minus_p_value_{window }'] = (std * z * (1 - adf_p))
                if D >= 252:
                    df[f'{prefix}_{r_or_p}_res_mom_12_1_{window }'] = res_mom_long_1
                else:
                    df[f'{prefix}_{r_or_p}_res_mom_06_1_{window }'] = res_mom_long_1
                df[f'{prefix}_{r_or_p}_res_rev_5d_{window }'] = res_rev_5d
                df[f'{prefix}_{r_or_p}_res_rsi_14_{window }'] = res_rsi_14
                df[f'{prefix}_{r_or_p}_res_streak_{window }'] = res_streak
                df[f'{prefix}_{r_or_p}_macd_line_{window }'] = macd_line
                df[f'{prefix}_{r_or_p}_macd_signal_{window }'] = macd_signal
                df[f'{prefix}_{r_or_p}_macd_hist_{window }'] = macd_hist
                df[f'{prefix}_{r_or_p}_resid_forecast_{window}'] = resid_forecast
                df[f'{prefix}_{r_or_p}_resid_forecast_sum_{window}'] = resid_forecast_sum

    return df


def build_other_mean_reversion_features(
    p: pd.Series,
    etf_p: pd.Series,
    sp500_p: pd.Series,
):
    D6 = 126
    D12 = 252
    D24 = 504

    p_r = np.log(p / p.shift(1)).copy()
    s_r = np.log(etf_p / etf_p.shift(1)).copy()
    sp_r = np.log(sp500_p / sp500_p.shift(1)).copy()

    sector_tracking_error_d6 = (p_r - s_r).rolling(D6).std()
    sector_tracking_error_d12 = (p_r - s_r).rolling(D12).std()
    sector_tracking_error_d24 = (p_r - s_r).rolling(D24).std()

    econ_tracking_error_d6 = (p_r - sp_r).rolling(D6).std()
    econ_tracking_error_d12 = (p_r - sp_r).rolling(D12).std()
    econ_tracking_error_d24 = (p_r - sp_r).rolling(D24).std()

    def variance_ratio(r: pd.Series, k: int) -> pd.Series:
        r_k = r.rolling(k).sum()
        var_1 = r.rolling(k).var(ddof=1)
        var_k = r_k.rolling(k).var(ddof=1)
        return var_k / (k * var_1)

    vr_5d = variance_ratio(p_r, 5)
    vr_20d = variance_ratio(p_r, 20)
    vr_60d = variance_ratio(p_r, 60)
    vr_120d = variance_ratio(p_r, 120)
    mean_reversion_df = pd.DataFrame({
        'Stock_price': p,
        'ETF_price': etf_p,
        'SP500_price': sp500_p,
        'Stock_return': p_r,
        'Sector_return': s_r,
        'Econ_return': sp_r,
        'vr_5d': vr_5d,
        'vr_20d': vr_20d,
        'vr_60d': vr_60d,
        'vr_120d': vr_120d,
        'sector_tracking_error_d1': sector_tracking_error_d6,
        'sector_tracking_error_d1': sector_tracking_error_d12,
        'sector_tracking_error_d1': sector_tracking_error_d24,
        'econ_tracking_error_d1': econ_tracking_error_d6,
        'econ_tracking_error_d1': econ_tracking_error_d12,
        'econ_tracking_error_d1': econ_tracking_error_d24,
    })
    return mean_reversion_df


def concat_mean_reversion_dataframes(
    list_of_dicts=None,
    selected_tickers=None,
) -> pd.DataFrame:
    if list_of_dicts is None:
        list_of_dicts = []
    if selected_tickers is None:
        selected_tickers = []

    all_ticker_dfs = []
    for ticker in selected_tickers:
        temp = pd.concat(
            [_dict[ticker] for _dict in list_of_dicts],
            axis=1,
        ).copy()
        temp["Ticker"] = ticker
        all_ticker_dfs.append(temp)

    ml_training_df = pd.concat(all_ticker_dfs, axis=0, ignore_index=False)
    return ml_training_df


PredictorConfig = tuple[pd.Series | pd.DataFrame, str, str]
# (X, sector_type, series_type)  e.g. (etf_p, 'sector', 'price'), (fama_df, 'fama', 'return')


def _align_return_predictors(
    return_target: pd.Series,
    predictor_configs: list[PredictorConfig],
) -> tuple[pd.Series, dict[tuple[str, str], pd.Series | pd.DataFrame]]:
    """
    Align the stock return target with every return-type predictor on one timeline.

    Returns aligned return_target and a lookup
    ``{(sector_type, series_type): aligned_X}`` for return configs only.
    """
    parts = [return_target.rename("_y")]
    keys: list[tuple[str, str]] = []

    for x, sector_type, series_type in predictor_configs:
        if series_type != "return":
            continue
        key = (sector_type, series_type)
        keys.append(key)
        if isinstance(x, pd.Series):
            parts.append(x.rename(sector_type))
        else:
            for col in x.columns:
                parts.append(x[col].rename(f"{sector_type}::{col}"))

    if len(parts) == 1:
        return return_target, {}

    aligned = pd.concat(parts, axis=1).dropna()
    aligned_target = aligned["_y"]
    aligned_x: dict[tuple[str, str], pd.Series | pd.DataFrame] = {}

    for x, sector_type, series_type in predictor_configs:
        if series_type != "return":
            continue
        key = (sector_type, series_type)
        if isinstance(x, pd.Series):
            aligned_x[key] = aligned[sector_type]
        else:
            cols = [f"{sector_type}::{c}" for c in x.columns]
            aligned_x[key] = aligned[cols].rename(columns=dict(zip(cols, x.columns)))

    return aligned_target, aligned_x


def compute_mean_reversion_for_ticker(
    ticker: str,
    price_target: pd.Series,
    return_target: pd.Series,
    predictor_configs: list[PredictorConfig],
    windows: list,
    existing_ticker_data: dict = None,
):
    """
    Stateless per-ticker worker for joblib.Parallel(backend='loky').

    Parameters
    ----------
    ticker : str
        Stock ticker.
    price_target : pd.Series
        Daily adjusted close for the stock.
    return_target : pd.Series
        Daily log returns for the stock (precomputed by caller).
    predictor_configs : list of (X, sector_type, series_type)
        Each entry is one predictor in the units it should be regressed on.
        ``series_type`` is ``'price'`` or ``'return'`` and selects which stock
        target series is used. Examples::

            [
                (etf_p, 'sector', 'price'),
                (etf_r, 'sector', 'return'),
                (sp500_p, 'econ', 'price'),
                (sp500_r, 'econ', 'return'),
                (fama_factors_df, 'fama', 'return'),  # return-only OK
            ]

        ``sector_type`` is naming-only (cache keys / output columns).
        Return predictors that are already returns (e.g. Fama) are passed as-is
        with ``series_type='return'``; no price counterpart is required.
    windows : list[int]
        Rolling window sizes (e.g. [126, 252, 504]).
    existing_ticker_data : dict or None
        Existing cache slice for this ticker.

    Returns
    -------
    ticker, updated_ticker_data, vannila_df
    """
    local: dict = {}
    if existing_ticker_data:
        local[ticker] = existing_ticker_data

    aligned_return_target, aligned_return_x = _align_return_predictors(
        return_target, predictor_configs)

    for window in windows:
        for x, sector_type, series_type in predictor_configs:
            if series_type == "price":
                fill_missing_mean_reversion_features(
                    local, price_target, x, ticker, "price", window, sector_type)
            elif series_type == "return":
                x_use = aligned_return_x.get((sector_type, series_type), x)
                fill_missing_mean_reversion_features(
                    local,
                    aligned_return_target,
                    x_use,
                    ticker,
                    "return",
                    window,
                    sector_type,
                )
            else:
                raise ValueError(
                    f"series_type must be 'price' or 'return', got {series_type!r}"
                )

    sector_price = next(
        (x for x, st, ser in predictor_configs if st == "sector" and ser == "price"),
        None,
    )
    econ_price = next(
        (x for x, st, ser in predictor_configs if st == "econ" and ser == "price"),
        None,
    )
    if sector_price is None or econ_price is None:
        raise ValueError(
            "predictor_configs must include ('sector', 'price') and ('econ', 'price') "
            "entries for build_other_mean_reversion_features."
        )
    vannila_df = build_other_mean_reversion_features(price_target, sector_price, econ_price)
    return ticker, local.get(ticker, {}), vannila_df


def compute_mean_reversion_variants_for_ticker(
    ticker: str,
    ticker_cache: dict,
    windows: list,
    sector_types: list[str],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Stateless per-ticker worker: materialize price + return variant DataFrames
    from one ticker's cache slice. Designed for joblib.Parallel(backend='loky').
    """
    cache_wrapper = {ticker: ticker_cache}
    price_df = create_mean_reversion_variants(
        cache_wrapper, ticker, windows, sector_types, is_price_series=True)
    return_df = create_mean_reversion_variants(
        cache_wrapper, ticker, windows, sector_types, is_price_series=False)
    return ticker, price_df, return_df
