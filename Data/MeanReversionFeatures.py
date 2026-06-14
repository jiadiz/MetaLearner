
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

    Mathematically equivalent (to float precision) to
    ``LinearRegression().fit(x.reshape(-1, 1), y)``, but skips all of
    sklearn's per-call estimator setup / input-validation overhead. That
    overhead dwarfs the actual linear-algebra work for the small windows
    used here (typ. 126 / 252 / 504), so this is the single largest
    speedup in the per-date hot loop.

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


def fit_ar1(y):
    """
    Fit AR(1): y_t = alpha + beta * y_{t-1} + eps.
    Returns (alpha, beta).

    Accepts a numpy array or pandas Series; NaNs are dropped before
    fitting (consistent with the previous sklearn-based implementation).
    Internally uses closed-form OLS via _simple_ols.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    y_arr = y_arr[~np.isnan(y_arr)]
    if len(y_arr) < 3:
        return np.nan, np.nan
    return _simple_ols(y_arr[:-1], y_arr[1:])

def forecast_k_steps_ar1(alpha: float,
                         beta: float,
                         y_t: float,
                         k: int = 21) -> float:

    # Handle beta ~ 1 (near unit root)
    if abs(1.0 - beta) < 1e-12:
        # Roughly: y_{t+k} ≈ y_t + k*alpha
        return y_t + k * alpha

    mu = alpha / (1.0 - beta)
    y_forecast = mu + (beta ** k) * (y_t - mu)
    return float(y_forecast)

def create_residual_mean_reversion_features(y_arr: np.ndarray,
                                            x_arr: np.ndarray):
    """
    One-feature OLS of y on x using all but the last observation
    (the "training window"), then derive residual-based diagnostics
    (std, ADF p-value, today's residual + z-score, AR(1)-based
    21-step-ahead residual forecast).

    Parameters
    ----------
    y_arr, x_arr : np.ndarray
        1-D numpy arrays of identical length (typically ``lookback + 1``).
        The last element is "today"; everything before is the regression
        window. NaNs are jointly dropped before fitting, mirroring the
        previous pd.Series.dropna() behavior but tolerant of NaNs in
        either array.

    Returns
    -------
    8-tuple of floats:
        (y_today, pair_beta, adf_pval, z_score_today, resid_std,
         resid_today, resid_forecast, resid_forecasted_change)
    """
    y_arr = np.asarray(y_arr, dtype=float).ravel()
    x_arr = np.asarray(x_arr, dtype=float).ravel()
    mask = ~(np.isnan(y_arr) | np.isnan(x_arr))
    y_arr = y_arr[mask]
    x_arr = x_arr[mask]

    if len(y_arr) < 3:
        nan = np.nan
        return nan, nan, nan, nan, nan, nan, nan, nan

    x_train = x_arr[:-1]
    y_train = y_arr[:-1]
    x_today = x_arr[-1]
    y_today = float(y_arr[-1])

    intercept, pair_beta = _simple_ols(x_train, y_train)
    if np.isnan(pair_beta):
        nan = np.nan
        return y_today, nan, nan, nan, nan, nan, nan, nan

    # residuals in the window (for std & ADF)
    residuals = y_train - (intercept + pair_beta * x_train)

    # std of residuals (sample std, ddof=1)
    std = float(residuals.std(ddof=1))

    # today's residual (one-step ahead) using today's x, y
    resid_today = float(y_today - (intercept + pair_beta * x_today))

    z_score_today = resid_today / std if std != 0.0 else np.nan

    # ADF p-value on window residuals (kept identical to prior version)
    try:
        adf_pval = float(adfuller(residuals, autolag='AIC')[1])
    except Exception:
        adf_pval = np.nan

    # AR(1)-based 21-step-ahead residual forecast
    ar1_alpha, ar1_beta = fit_ar1(residuals)
    resid_forecast = forecast_k_steps_ar1(ar1_alpha,
                                          ar1_beta,
                                          resid_today,
                                          k=21 + 1)
    resid_forecasted_change = resid_forecast - resid_today

    return (y_today, float(pair_beta), adf_pval, z_score_today, std,
            resid_today, resid_forecast, resid_forecasted_change)


def fill_missing_mean_reversion_features(available_mean_reversion_features_per_ticker: dict,
                                         p: pd.Series,
                                         etf_p : pd.Series, 
                                         sp500_p: pd.Series,
                                         ticker: str,
                                         series_type: str,
                                         lookback: int,
                                         verbose: bool = False,
                                         ):

    def identify_ticker_feature_location(available_mean_reversion_features_per_ticker, ticker, mean_reversion_type):
        if ticker not in available_mean_reversion_features_per_ticker:
            available_mean_reversion_features_per_ticker[ticker] = {}
            if verbose:
                print(f'{ticker} data created')
        if  mean_reversion_type not in available_mean_reversion_features_per_ticker[ticker]:
            available_mean_reversion_features_per_ticker[ticker][mean_reversion_type] = {}
            if verbose:
                print(f'{ticker} {mean_reversion_type} data created')
        data = available_mean_reversion_features_per_ticker[ticker][mean_reversion_type]
        return data

    if series_type == 'return':
        p = np.log(p / p.shift(1)).copy()
        etf_p = np.log(etf_p / etf_p.shift(1)).copy()
        sp500_p = np.log(sp500_p / sp500_p.shift(1)).copy()

        # Align all return series to one shared valid timeline.
        common_idx = p.dropna().index.intersection(etf_p.dropna().index).intersection(sp500_p.dropna().index)
        p = p.loc[common_idx]
        etf_p = etf_p.loc[common_idx]
        sp500_p = sp500_p.loc[common_idx]

    # Convert to raw numpy once. The per-date hot loops below slice these
    # arrays as O(1) views instead of building a fresh pandas Series +
    # DatetimeIndex slice per date.
    p_arr     = p.values
    etf_arr   = etf_p.values
    sp500_arr = sp500_p.values
    dates     = p.index

    ticker = ticker

    sector_type = 'sector'

    mean_reversion_type = f'mean_reversion_{sector_type}_{series_type}_d{lookback}'

    data = identify_ticker_feature_location(available_mean_reversion_features_per_ticker, ticker, mean_reversion_type)

    # Iterate positionally to avoid an O(log n)-ish get_loc per date.
    for pos, date in enumerate(p.index):
        # Backfill older cached entries that were written before resid_forecast was stored.
        existing = data.get(date)
        needs_backfill = (
            isinstance(existing, dict)
            and 'y_today' in existing
            and 'resid_forecast' not in existing
        )
        if (date not in data) or needs_backfill:
            start = max(0, pos - lookback)
            y_win = p_arr[start: pos + 1]
            x_win = etf_arr[start: pos + 1]
            if len(y_win) < lookback + 1:
                continue
            (y_today, beta, adf_p, z_score_today, resid_std,
             resid_today, resid_forecast,
             resid_forecasted_change) = create_residual_mean_reversion_features(y_win, x_win)

            # Single dict allocation with all fields populated up-front is cheaper
            # than `data[date] = {}` followed by eight `data[date][k] = v` assigns.
            data[date] = {
                'y_today': y_today,
                'beta': beta,
                'adf_p': adf_p,
                'z_score_today': z_score_today,
                'resid_std': resid_std,
                'resid_today': resid_today,
                'resid_forecast': resid_forecast,
                'resid_forecasted_change': resid_forecasted_change,
            }
    if verbose:
        print('Data until ', p.index[-1], f'{mean_reversion_type} data created')

    sector_type = 'econ'

    mean_reversion_type = f'mean_reversion_{sector_type}_{series_type}_d{lookback}'

    data = identify_ticker_feature_location(available_mean_reversion_features_per_ticker, ticker, mean_reversion_type)

    for pos, date in enumerate(p.index):
        existing = data.get(date)
        needs_backfill = (
            isinstance(existing, dict)
            and 'y_today' in existing
            and 'resid_forecast' not in existing
        )
        if (date not in data) or needs_backfill:
            start = max(0, pos - lookback)
            y_win = p_arr[start: pos + 1]
            x_win = sp500_arr[start: pos + 1]
            if len(y_win) < lookback + 1:
                continue

            (y_today, beta, adf_p, z_score_today, resid_std,
             resid_today, resid_forecast,
             resid_forecasted_change) = create_residual_mean_reversion_features(y_win, x_win)

            data[date] = {
                'y_today': y_today,
                'beta': beta,
                'adf_p': adf_p,
                'z_score_today': z_score_today,
                'resid_std': resid_std,
                'resid_today': resid_today,
                'resid_forecast': resid_forecast,
                'resid_forecasted_change': resid_forecasted_change,
            }
    if verbose:
        print('Data until ', p.index[-1], f'{mean_reversion_type} data created')
# create_residual_mean_reversion_features(np.log(price_data['MMM_Close'] / price_data['MMM_Close'].shift(1)).copy(),
#                                         np.log(price_data['XLI_Close'] / price_data['XLI_Close'].shift(1)).copy())

def create_mean_reversion_variants(available_mean_reversion_features_per_ticker: dict, 
                                   ticker, windows, is_price_series = True):

    if is_price_series == True:
        return_or_price = 'price'
    else:
        return_or_price = 'return'

    # ticker_base_variables = available_mean_reversion_features_per_ticker['MMM']

    df = pd.DataFrame()

    for econ_or_sector in ['sector', 'econ']:
        for window in windows:

            ys = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window}']).apply(lambda x: x['y_today'] if 'y_today' in x else np.nan)

            beta = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window}']).apply(lambda x: x['beta'] if 'beta' in x else np.nan)

            z = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window }']).apply(lambda x: x['z_score_today'] if 'z_score_today' in x else np.nan)

            adf_p = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window }']).apply(lambda x: x['adf_p'] if 'adf_p' in x else np.nan)

            std = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window }']).apply(lambda x: x['resid_std'] if 'resid_std' in x else np.nan)

            resid = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window }']).apply(lambda x: x['resid_today'] if 'resid_today' in x else np.nan)

            # AR(1)-based 21-step-ahead residual forecast (matches ground truth notebook).
            resid_forecast = pd.Series(available_mean_reversion_features_per_ticker[ticker][f'mean_reversion_{econ_or_sector}_{return_or_price}_d{window }']).apply(lambda x: x['resid_forecast'] if isinstance(x, dict) and 'resid_forecast' in x else np.nan)

            # print(adf_p)

            if window >= 252:
                MOM_12_LEN   = 252
            else:
                MOM_06_LEN   = 126 - 21

            MOM_12_SKIP  = 21      # skip last ~1 month (12–1)
            MOM_3M_LEN   = 63      # ~3 months
            REV_5D_LEN   = 5
            RSI_LEN      = 14

            MACD_FAST    = 12
            MACD_SLOW    = 26
            MACD_SIGNAL  = 9
            alpha_fast   = 2 / (MACD_FAST + 1)
            alpha_slow   = 2 / (MACD_SLOW + 1)
            alpha_sig    = 2 / (MACD_SIGNAL + 1)

            if window >= 252:
                res_mom_12_1 = resid.expanding().apply(lambda x: residual_momentum(x, length=MOM_12_LEN, skip=MOM_12_SKIP))
                # Long windows should use 12-1 residual momentum.
                res_mom_long_1 = res_mom_12_1

            else:
                res_mom_06_1 = resid.expanding().apply(lambda x: residual_momentum(x, length=MOM_06_LEN, skip=MOM_12_SKIP))
                res_mom_long_1 = res_mom_06_1

            res_rsi_14 = resid.expanding().apply(lambda x: residual_rsi(x, RSI_LEN ), raw=False)


            # 3-month residual momentum
            res_mom_3m = resid.expanding(MOM_3M_LEN+1).apply(lambda x: residual_momentum(x, length=MOM_3M_LEN,skip=0))

            # 5-day residual reversal
            res_rev_5d = resid.expanding().apply(lambda x: residual_reversal(x, length=REV_5D_LEN))

            res_streak = resid.iloc[window:].expanding().apply(lambda x: residual_streak_length(x)).reindex(resid.index)

            ema_fast = resid.iloc[window:].ewm(span=MACD_FAST, adjust=False).mean().reindex(resid.index)
            ema_slow = resid.iloc[window:].ewm(span=MACD_SLOW, adjust=False).mean().reindex(resid.index)

            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
            macd_hist = macd_line - macd_signal

            beta, z, adf_p, std, resid

            if is_price_series == True:
                stock_price = ys
                r_or_p = 'p'
            else:
                r_or_p = 'r'

            if is_price_series == True:

                D = window
                # df['Stock_price'] = ys
                df[f'{econ_or_sector}_{r_or_p}_beta_d{window}'] = beta

                df[f'{econ_or_sector}_{r_or_p}_resid_std_d{window }'] = std/stock_price 
                df[f'{econ_or_sector}_{r_or_p}_resid_z_d{window }'] = z

                df[f'{econ_or_sector}_{r_or_p}_resid_total_deviation_per_price_d{window}'] = (std*z)/stock_price 
                # if r_or_p == 'p' and window == 126:
                #     print(df[f'{econ_or_sector}_{r_or_p}_resid_total_deviation_per_price'])
                #     print(df[f'{econ_or_sector}_{r_or_p}_resid_std_d{window }'] *df[f'{econ_or_sector}_{r_or_p}_resid_z_d{window }'])

                df[f'{econ_or_sector}_{r_or_p}_resid_adf_p_d{window }'] = adf_p
                df[f'{econ_or_sector}_{r_or_p}_z_times_1_minus_p_value_{window }'] = z * (1 - adf_p)
                df[f'{econ_or_sector}_{r_or_p}_resid_std_times_z_times_1_minus_p_value_{window }'] = (std * z * (1 - adf_p))/ stock_price 
                if D >= 252:
                    df[f'{econ_or_sector}_{r_or_p}_res_mom_12_1_{window }'] = res_mom_long_1/stock_price
                else:
                    df[f'{econ_or_sector}_{r_or_p}_res_mom_06_1_{window }'] = res_mom_long_1/stock_price
                df[f'{econ_or_sector}_{r_or_p}_res_rev_5d_{window }'] = res_rev_5d/stock_price
                df[f'{econ_or_sector}_{r_or_p}_res_rsi_14_{window }'] = (res_rsi_14*std*z*adf_p)/stock_price
                df[f'{econ_or_sector}_{r_or_p}_res_streak_{window }'] = (res_streak*std*z*adf_p)/stock_price
                df[f'{econ_or_sector}_{r_or_p}_macd_line_{window }'] = macd_line/stock_price
                df[f'{econ_or_sector}_{r_or_p}_macd_signal_{window }'] = macd_signal/stock_price
                df[f'{econ_or_sector}_{r_or_p}_macd_hist_{window }'] = macd_hist/stock_price
                # 21-step AR(1) residual forecast, price-adjusted to match ground truth notebook.
                df[f'{econ_or_sector}_{r_or_p}_resid_forecast_{window}'] = resid_forecast / stock_price
            elif is_price_series == False:

                D = window

                df[f'{econ_or_sector}_{r_or_p}_beta_d{window}'] = beta

                df[f'{econ_or_sector}_{r_or_p}_resid_std_d{window }'] = std
                df[f'{econ_or_sector}_{r_or_p}_resid_z_d{window }'] = z

                df[f'{econ_or_sector}_{r_or_p}_resid_total_deviation_per_price_d{window}'] = (std*z)
                # if r_or_p == 'p' and window == 126:
                #     print(df[f'{econ_or_sector}_{r_or_p}_resid_total_deviation_per_price'])
                #     print(df[f'{econ_or_sector}_{r_or_p}_resid_std_d{window }'] *df[f'{econ_or_sector}_{r_or_p}_resid_z_d{window }'])

                df[f'{econ_or_sector}_{r_or_p}_resid_adf_p_d{window }'] = adf_p
                df[f'{econ_or_sector}_{r_or_p}_z_times_1_minus_p_value_{window }'] = z * (1 - adf_p)
                df[f'{econ_or_sector}_{r_or_p}_resid_std_times_z_times_1_minus_p_value_{window }'] = (std * z * (1 - adf_p))
                if D >= 252:
                    df[f'{econ_or_sector}_{r_or_p}_res_mom_12_1_{window }'] = res_mom_long_1
                else:
                    df[f'{econ_or_sector}_{r_or_p}_res_mom_06_1_{window }'] = res_mom_long_1
                df[f'{econ_or_sector}_{r_or_p}_res_rev_5d_{window }'] = res_rev_5d
                df[f'{econ_or_sector}_{r_or_p}_res_rsi_14_{window }'] = res_rsi_14
                df[f'{econ_or_sector}_{r_or_p}_res_streak_{window }'] = res_streak
                df[f'{econ_or_sector}_{r_or_p}_macd_line_{window }'] = macd_line
                df[f'{econ_or_sector}_{r_or_p}_macd_signal_{window }'] = macd_signal
                df[f'{econ_or_sector}_{r_or_p}_macd_hist_{window }'] = macd_hist
                # 21-step AR(1) residual forecast in return-space (matches ground truth notebook).
                df[f'{econ_or_sector}_{r_or_p}_resid_forecast_{window}'] = resid_forecast

    return df

def build_other_mean_reversion_features(p: pd.Series,
                                         etf_p : pd.Series, 
                                         sp500_p: pd.Series,
                                         ):
       # Horizons (trading days)

    # D1  = 21       # ~1 month
    # D3  = 63       # ~3 months
    D6  = 126      # ~6 months
    D12 = 252      # ~12 months
    D24 = 504


    p_r =  np.log(p / p.shift(1)).copy()
    s_r =  np.log(etf_p / etf_p.shift(1)).copy()
    sp_r =  np.log(sp500_p / sp500_p.shift(1)).copy()

    # sector_tracking_error_d1 = (p_r - s_r).rolling(D1).std()
    # sector_tracking_error_d3 = (p_r - s_r).rolling(D3).std()
    sector_tracking_error_d6 = (p_r - s_r).rolling(D6).std()
    sector_tracking_error_d12 = (p_r - s_r).rolling(D12).std()
    sector_tracking_error_d24 = (p_r - s_r).rolling(D24).std()

    # econ_tracking_error_d1 = (p_r - sp_r).rolling(D1).std()
    # econ_tracking_error_d3 = (p_r - sp_r).rolling(D3).std()
    econ_tracking_error_d6 = (p_r - sp_r).rolling(D6).std()
    econ_tracking_error_d12 = (p_r - sp_r).rolling(D12).std()
    econ_tracking_error_d24 = (p_r - sp_r).rolling(D24).std()

    from sklearn.linear_model import LinearRegression


    def variance_ratio(r: pd.Series, k: int) -> pd.Series:
        """
        Rolling variance ratio VR(k) = Var(k-day returns) / (k * Var(1-day returns)),
        both variances estimated over a rolling window of length k.
        """
        # k-day cumulative return (sum of daily returns)
        r_k = r.rolling(k).sum()

        # Rolling variance of 1-day returns over window k
        var_1 = r.rolling(k).var(ddof=1)

        # Rolling variance of k-day returns over window k
        var_k = r_k.rolling(k).var(ddof=1)

        vr = var_k / (k * var_1)
        return vr

    # Assuming r is a pandas Series of daily returns indexed by date
    vr_5d  = variance_ratio(p_r, 5)
    vr_20d = variance_ratio(p_r, 20)
    vr_60d = variance_ratio(p_r, 60)
    vr_120d = variance_ratio(p_r, 120)
    mean_reversion_df = pd.DataFrame({'Stock_price':p,
                       'ETF_price':etf_p,
                       'SP500_price':sp500_p,
                        'Stock_return':p_r,
                       'Sector_return':s_r,
                       'Econ_return':sp_r,
                       'vr_5d':vr_5d,
                       'vr_20d':vr_20d,
                       'vr_60d':vr_60d,
                       'vr_120d':vr_120d,
                    # 'sector_tracking_error_d1':sector_tracking_error_d1,
                       # 'sector_tracking_error_d1':sector_tracking_error_d3,
                       'sector_tracking_error_d1':sector_tracking_error_d6,
                       'sector_tracking_error_d1':sector_tracking_error_d12,
                       'sector_tracking_error_d1':sector_tracking_error_d24,
                       # 'econ_tracking_error_d1':econ_tracking_error_d1,
                       # 'econ_tracking_error_d1':econ_tracking_error_d3,
                       'econ_tracking_error_d1':econ_tracking_error_d6,
                       'econ_tracking_error_d1':econ_tracking_error_d12,
                       'econ_tracking_error_d1':econ_tracking_error_d24,
                      })
    return mean_reversion_df


def concat_mean_reversion_dataframes(
    list_of_dicts = [],
    selected_tickers = []
) -> pd.DataFrame:
    
    all_ticker_dfs = []

    for ticker in selected_tickers:
        temp = pd.concat(
            [
                _dict[ticker] for _dict in list_of_dicts
            ],
            axis=1
        ).copy()

        temp["Ticker"] = ticker
        all_ticker_dfs.append(temp)

    ml_training_df = pd.concat(all_ticker_dfs, axis=0, ignore_index=False)
    return ml_training_df


def compute_mean_reversion_for_ticker(
    ticker: str,
    p: pd.Series,
    etf_p: pd.Series,
    sp500_p: pd.Series,
    windows: list,
    existing_ticker_data: dict = None,
):
    """
    Stateless per-ticker worker designed for joblib.Parallel(backend='loky').

    Takes the optional existing cache slice for `ticker` (typically
    available_mean_reversion_features_per_ticker.get(ticker)), fills in any
    dates that are missing for each (window, series_type) pair via the
    existing fill_missing_mean_reversion_features logic, and returns the
    updated slice along with the 'vannila' mean reversion DataFrame.

    Parameters
    ----------
    ticker : str
        The stock ticker.
    p, etf_p, sp500_p : pd.Series
        Daily price series for the stock, its sector ETF, and the S&P 500
        proxy. Indexed by date.
    windows : list[int]
        Rolling window sizes to populate (e.g. [126, 252, 504]).
    existing_ticker_data : dict or None
        The existing cache slice for this ticker. Pass None or {} for a
        first-time / cold-cache computation. May be mutated in-place; the
        caller should pass a copy if it needs to preserve the original.
        Under loky, the input is already pickled across the process
        boundary so in-place mutation is harmless to the parent.

    Returns
    -------
    ticker : str
        Echoed back to make merging in the parent unambiguous.
    updated_ticker_data : dict
        New cache slice. Caller should set
        ``available_mean_reversion_features_per_ticker[ticker] = updated_ticker_data``.
    vannila_df : pd.DataFrame
        Output of build_other_mean_reversion_features(p, etf_p, sp500_p).
    """
    local: dict = {}
    if existing_ticker_data:
        local[ticker] = existing_ticker_data

    for window in windows:
        fill_missing_mean_reversion_features(
            local, p, etf_p, sp500_p, ticker, 'price', window
        )
        fill_missing_mean_reversion_features(
            local, p, etf_p, sp500_p, ticker, 'return', window
        )

    vannila_df = build_other_mean_reversion_features(p, etf_p, sp500_p)
    return ticker, local.get(ticker, {}), vannila_df
