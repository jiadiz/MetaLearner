

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

def build_features(price: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """
    price: daily close price series with a DatetimeIndex
    volume: optional daily volume series aligned with price.index
    Returns (data, last_row_features_without_y)
    - X: engineered features using ONLY past data at each t
    - y: next 21D log return
    """
    import numpy as np
    import pandas as pd

    s = price.astype(float).copy()
    s.index = pd.to_datetime(s.index).sort_values()
    # daily log returns
    r = np.log(s / s.shift(1))

    # Horizons (trading days)
    D1  = 21       # ~1 month
    D3  = 63       # ~3 months
    D6  = 126      # ~6 months
    D12 = 252      # ~12 months

    # -------------------------
    # Helpers
    # -------------------------
    def rolling_autocorr(series: pd.Series, win: int, lag: int) -> pd.Series:
        def _ac(vals):
            if len(vals) <= lag:
                return np.nan
            x1, x2 = vals[:-lag], vals[lag:]
            if np.std(x1) == 0 or np.std(x2) == 0:
                return np.nan
            return np.corrcoef(x1, x2)[0, 1]
        return series.rolling(win).apply(_ac, raw=True)

    def rolling_percentile_rank(series: pd.Series, win: int) -> pd.Series:
        def _prc(vals):
            last = vals[-1]
            return (np.sum(vals <= last) / len(vals)) if len(vals) > 0 else np.nan
        return series.rolling(win).apply(_prc, raw=True)

    def rolling_trend_slope(log_price: pd.Series, win: int) -> pd.Series:
        idx = np.arange(win, dtype=float)
        x = idx - idx.mean()
        denom = np.sum(x**2)
        def _slope(vals):
            if len(vals) < win:
                return np.nan
            y = vals - vals.mean()
            beta = np.sum(x * y) / denom
            return beta * 252.0   # annualized
        return log_price.rolling(win).apply(_slope, raw=True)

    # -------------------------
    # Momentum & returns
    # -------------------------
    mom_3_1  = r.shift(D1).rolling(D3).sum()
    mom_6_1  = r.shift(D1).rolling(D6).sum()
    mom_12_1 = r.shift(D1).rolling(D12).sum()
    ret_1m   = r.rolling(D1).sum()

    # Volatility
    vol_1m = r.rolling(D1).std()
    vol_3m = r.rolling(D3).std()
    vol_6m = r.rolling(D6).std()

    # Moving-average gap
    ma_50  = s.rolling(50).mean()
    ma_200 = s.rolling(200).mean()
    ma_gap_50_200 = (ma_50 / ma_200) - 1.0

    # RSI(14)
    diff = s.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    RSI_14 = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12,26) & signal
    ema12 = s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = s.ewm(span=26, adjust=False, min_periods=26).mean()
    MACD = ema12 - ema26
    MACD_signal = MACD.ewm(span=9, adjust=False, min_periods=9).mean()
    MACD_hist = MACD - MACD_signal

    # Bollinger %B
    bb_mid = s.rolling(20).mean()
    bb_std = s.rolling(20).std()
    BB_pctB = (s - (bb_mid - 2*bb_std)) / ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std))

    # Skewness & kurtosis of returns
    ret_skew_1m = r.rolling(D1).skew()
    ret_kurt_1m = r.rolling(D1).kurt()
    ret_skew_3m = r.rolling(D3).skew()
    ret_kurt_3m = r.rolling(D3).kurt()

    # Autocorrelation of returns
    ac1_3m  = rolling_autocorr(r, D3, 1)
    ac5_3m  = rolling_autocorr(r, D3, 5)
    ac21_3m = rolling_autocorr(r, D3, 21)

    # Drawdowns
    cum_max = s.cummax()
    drawdown = (s / cum_max) - 1.0
    mdd_3m = drawdown.rolling(D3).min()
    mdd_6m = drawdown.rolling(D6).min()

    # 52w high/low distance & percentile rank
    roll_max_252 = s.rolling(D12).max()
    roll_min_252 = s.rolling(D12).min()
    dist_52w_high = (s / roll_max_252) - 1.0
    dist_52w_low  = (s / roll_min_252) - 1.0
    prc_rank_252  = rolling_percentile_rank(s, D12)

    # Trend slope on log-price
    log_s = np.log(s.replace(0, np.nan))
    slope_log_3m = rolling_trend_slope(log_s, D3)
    slope_log_6m = rolling_trend_slope(log_s, D6)

    # Vol-of-vol
    vol_of_vol_3m = vol_1m.rolling(D3).std()

    # -------------------------
    # Volume-based features
    # -------------------------
    if volume is not None:
        v = volume.reindex(s.index).astype(float)

        # OBV
        direction = np.sign(s.diff())
        OBV = (direction * v).fillna(0).cumsum()

        # Up/Down volume ratio (21d)
        up_vol = v.where(s.diff() > 0, 0).rolling(D1).sum()
        down_vol = v.where(s.diff() < 0, 0).rolling(D1).sum()
        up_down_vol_ratio = up_vol / down_vol.replace(0, np.nan)

        # Volume spike flag (ratio > 2 vs 20d mean)
        vol_mean_20 = v.rolling(20).mean()
        vol_spike = (v / vol_mean_20 > 2.0).astype(int)
    else:
        OBV = up_down_vol_ratio = vol_spike = pd.Series(index=s.index, dtype=float)

    # -------------------------
    # Target (next 21D log return)
    # -------------------------
    y_next_1m = np.log(s.shift(-D1) / s)

    # -------------------------
    # Assemble
    # -------------------------
    X = pd.DataFrame({
        "mom_12_1": mom_12_1,
        "mom_6_1": mom_6_1,
        "mom_3_1": mom_3_1,
        "ret_1m": ret_1m,
        "vol_1m": vol_1m,
        "vol_3m": vol_3m,
        "vol_6m": vol_6m,
        "ma_gap_50_200": ma_gap_50_200,
        "RSI_14": RSI_14,
        "MACD": MACD,
        "MACD_signal": MACD_signal,
        "MACD_hist": MACD_hist,
        "BB_pctB_20": BB_pctB,
        "ret_skew_1m": ret_skew_1m,
        "ret_kurt_1m": ret_kurt_1m,
        "ret_skew_3m": ret_skew_3m,
        "ret_kurt_3m": ret_kurt_3m,
        "ac1_3m": ac1_3m,
        "ac5_3m": ac5_3m,
        "ac21_3m": ac21_3m,
        "drawdown_now": drawdown,
        "mdd_3m": mdd_3m,
        "mdd_6m": mdd_6m,
        "dist_52w_high": dist_52w_high,
        "dist_52w_low": dist_52w_low,
        "prc_rank_252": prc_rank_252,
        "slope_log_3m_ann": slope_log_3m,
        "slope_log_6m_ann": slope_log_6m,
        "vol_of_vol_3m": vol_of_vol_3m,
        # Volume-based
        "OBV": OBV,
        "up_down_vol_ratio_1m": up_down_vol_ratio,
        "vol_spike_flag": vol_spike,
    }, index=s.index)

    data = X.copy()
    data["y"] = y_next_1m

    # Latest row for inference (drop y)
    last_row = data.iloc[[-1]].drop(columns=["y"], errors="ignore")

    # Drop NaNs for training
    # data = data.dropna()

    return data, last_row


def add_sector(data_to_append_on:pd.DataFrame,
              sector_df:pd.DataFrame):
    if 'Sector' in data_to_append_on.columns:
        return data_to_append_on
    df = data_to_append_on.merge(sector_df, on = 'Ticker', how = 'left')
    return df

def append_features(data_to_append_on:pd.DataFrame,
              features:pd.DataFrame):
    if features.columns.isin(data_to_append_on).all():
        return data_to_append_on
    else:
        existing_columns = [i for i in features.columns if i in data_to_append_on.columns and i not in ['Ticker', 'Date']]
        data_to_append_on = data_to_append_on.drop(existing_columns, axis = 1) 
    df = data_to_append_on.merge(features, on = ['Ticker', 'Date'], how = 'left')
    return df


def create_ranking_df(price_data):
    price_columns = [i for i in price_data.columns if 'Close' in i]
    if 'Date' not in price_data.columns:
        price_data = price_data.reset_index().rename(columns = {'index':'Date'})
    price_only_data = price_data[['Date']+price_columns]

    price_only_data  = price_only_data .sort_values('Date')
    temp_price_only_data  = price_only_data.set_index('Date').copy()
    
    # Trading-day approximations
    MONTHS = {
        '1M': 21,   # ~1 month
        '3M': 63,   # ~3 months
        '6M': 126   # ~6 months
    }
    
    
    ranks_dfs = {}
    
    for label, window in MONTHS.items():
        # Compute forward-looking return: (today_price / price_n_days_ago) - 1
        ret = temp_price_only_data .pct_change(periods=window)
        rank = ret.rank(axis=1, ascending=False, method='min')

        rank.columns = [i.replace('Close', f'{label}_return_rank') for i in rank.columns]
    
        rank = (rank
             .reset_index()                                # expose Date
             .melt(id_vars='Date', var_name='col', value_name='rank'))

        rank['Ticker'] = rank['col'].str.split('_').str[0]
        rank['Window'] = rank['col'].str.split('_').str[1]
        rank = rank.drop('col', axis = 1)
        ranks_dfs[label] = rank

    # Unpack nicely
    rank_1m = ranks_dfs['1M']
    rank_3m = ranks_dfs['3M']
    rank_6m = ranks_dfs['6M']

    ranking_df = pd.concat([rank_1m, rank_3m, rank_6m])

    ranking_df = ranking_df.pivot(index = ['Date', 'Ticker'], columns = 'Window', values = 'rank').reset_index()
    ranking_df = ranking_df.rename(columns = {'1M':'1M_rank', '3M':'3M_rank', '6M':'6M_rank'})
    return ranking_df

def build_base_features_for_one_ticker(ticker, price_data, mapping, sector_to_etf):

    price_column  = f'{ticker}_Close'
    volume_column = f'{ticker}_Volume'

    # --- Price features ---
    feats_p, last_p = build_features(price_data[price_column].astype(float))
    feats_p = feats_p.reset_index().rename(columns={'index': 'Date'})
    last_p  = last_p.reset_index().rename(columns={'index': 'Date'})

    # --- Volume features ---
    feats_v, last_v = build_features(price_data[volume_column].astype(float))
    feats_v = feats_v.drop(columns=['y']).add_prefix('vol_').reset_index().rename(columns={'index': 'Date'})
    last_v  = last_v.add_prefix('vol_').reset_index().rename(columns={'index': 'Date'})

    # --- Merge stock price + volume features ---
    feats_i = feats_p.merge(feats_v, on='Date', how='left')
    feats_i['Ticker'] = ticker
    last_i = last_p.merge(last_v, on='Date', how='left')
    last_i['Ticker'] = ticker

    # --- Sector ETF features ---
    sector = mapping.loc[mapping['Ticker'] == ticker, 'Sector']
    sector = sector.iloc[0] if not sector.empty else np.nan
    etf_ticker = sector_to_etf.get(sector, "SPY")

    etf_price_col = f"{etf_ticker}_Close"
    etf_vol_col   = f"{etf_ticker}_Volume"

    if etf_price_col in price_data.columns and etf_vol_col in price_data.columns:
        etf_p, etf_last_p = build_features(price_data[etf_price_col].astype(float))
        etf_p = etf_p.drop(columns=['y']).add_prefix("etf_").reset_index().rename(columns={"index": "Date"})
        etf_last_p = etf_last_p.add_prefix("etf_").reset_index().rename(columns={"index": "Date"})

        etf_v, etf_last_v = build_features(price_data[etf_vol_col].astype(float))
        etf_v = etf_v.drop(columns=['y']).add_prefix("etf_vol_").reset_index().rename(columns={"index": "Date"})
        etf_last_v = etf_last_v.add_prefix("etf_vol_").reset_index().rename(columns={"index": "Date"})

        feats_i = feats_i.merge(etf_p, on="Date", how="left").merge(etf_v, on="Date", how="left")
        last_i  = last_i.merge(etf_last_p, on="Date", how="left").merge(etf_last_v, on="Date", how="left")


    # --- Mean reversion features ---

    return feats_i, last_i


def create_rolling_1m_volume_data(price_data):
    
    price_data_temp = price_data.copy()
    if 'Date' not in price_data.columns:
        price_data_temp = price_data_temp.reset_index()
    price_data_temp ['Date'] = pd.to_datetime(price_data_temp ['Date'])
    price_data_temp = price_data_temp.sort_values('Date')
    
    # identify volume columns
    volume_cols = [c for c in price_data_temp.columns if c.endswith('_Volume')]
    
    dfs = []
    
    for col in volume_cols:
        ticker = col.replace('_Volume', '')
    
        vol_col = col
        close_price_col = f'{ticker}_Close'
    
        
        df_tmp = (
            price_data_temp[['Date', vol_col, close_price_col]]
            .assign(
                ticker_name=ticker,
                rolling_1m_dollar_volume=lambda x: (x[vol_col]*x[close_price_col]).rolling(window=21, min_periods=1).sum()
            )
            .loc[:, ['Date', 'ticker_name', 'rolling_1m_dollar_volume']]
        )
        
        dfs.append(df_tmp)
    
    # concat vertically
    rolling_1m_dollar_volume_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    rolling_1m_dollar_volume_df= rolling_1m_dollar_volume_df .rename(columns = {'ticker_name':'Ticker'})
    return rolling_1m_dollar_volume_df

def append_volume_data(df, volume_df):
    if 'rolling_1m_dollar_volume' in df:
        df = df.drop('rolling_1m_dollar_volume', axis = 1)
    else:
        pass
    df = df.merge(volume_df, on = ['Date', 'Ticker'])
    return df