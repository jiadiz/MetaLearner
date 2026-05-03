
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

def create_lag_return(p: pd.Series, 
                      lookback: int) -> pd.Series:
    ret_lag = (p - p.shift(lookback)) / p.shift(lookback)
    return ret_lag

def create_fut_return(p: pd.Series, holddays: int) -> pd.Series:
    ret_fut = (p.shift(-holddays) - p) / p
    return ret_fut

def create_auto_correlation_forecast(ret_lag: pd.Series, 
                                     ret_fut: pd.Series, 
                                     lookback: int,
                                    holddays: int,
                                    min_n_indep: int):
    all_good = (~ret_lag.isna())& (~ret_fut.isna())
    ret_lag_train, ret_fut_train = ret_lag[all_good], ret_fut[all_good]

    if holddays <= lookback:
        idx = np.arange(0, len(ret_lag_train), holddays)
    else:
        idx = np.arange(0, len(ret_lag_train), lookback)

    ret_lag_train = ret_lag_train.iloc[idx]
    ret_fut_train = ret_fut_train.iloc[idx]
    # print(idx)
    if len(ret_lag_train) < min_n_indep:
        return np.nan, np.nan
    if ret_lag_train.std(ddof=1) == 0 or ret_fut_train.std(ddof=1) == 0:
        return np.nan, np.nan

    r, pv = pearsonr(ret_lag_train.to_numpy(), ret_fut_train .to_numpy())
    return r, pv


available_AR_values_per_stock = {}


def fill_missing_correlation_coefficients(available_AR_values_per_stock : dict,
                                          p: pd.Series,
                                          ticker: str,
                                         lookback: int,
                                          holddays: int
                                         ):

    ret_lag = create_lag_return(p, lookback)
    ret_fut = create_fut_return(p, holddays)

    ticker = ticker
    lag_fut_combo = f'lag_{lookback}_hold_{holddays}'

    if ticker not in available_AR_values_per_stock:
        available_AR_values_per_stock[ticker] = {}

    if lag_fut_combo not in available_AR_values_per_stock[ticker]:
        available_AR_values_per_stock[ticker][lag_fut_combo] = {}

    data = available_AR_values_per_stock[ticker][lag_fut_combo]

    # for date in p.index:
    for i, date in enumerate(p.index):
        if date not in data:
            # Match original notebook logic: train strictly before current date.
            # ret_lag_sub = ret_lag.loc[:date]
            # ret_fut_sub = ret_fut.loc[:date]
            ret_lag_sub = ret_lag.iloc[:i]
            ret_fut_sub = ret_fut.iloc[:i]

            r, pv = create_auto_correlation_forecast(ret_lag_sub, ret_fut_sub, 
                                     lookback, holddays,3)

            # data[date] = [r, pv] 
            
            # Match original notebook: forecast_ret_{lookback} = corr * ret_lag_at_trade_date
            ret_lag_today = ret_lag.iloc[i] if i < len(ret_lag) else np.nan
            forecast_ret_today = r * ret_lag_today if not (np.isnan(r) or np.isnan(ret_lag_today)) else np.nan
            data[date] = [forecast_ret_today, pv]
        
            


def create_compressed_forecast_features(temp_feat_df):
    df = temp_feat_df.copy()
    
    forecast_features = [i for i in df.columns if 'forecast_ret_' in i]
    short_term_forecast_features = [i for i in forecast_features if '_5' in i or '_5' in i or '_10' in i or '_25' in i]
    long_term_forecast_features = [i for i in forecast_features if '_60' in i or '_120' in i or '_250 ' in i]
    # scaler = StandardScaler()
    
    # forecast_features_data = scaler.fit_transform(temp_feat_df[forecast_features])
    average_forecasted_returns_of_momentums = df[forecast_features].mean(axis = 1)

    short_forecasted_returns_of_momentums = df[short_term_forecast_features].mean(axis = 1)

    long_forecasted_returns_of_momentums = df[long_term_forecast_features].mean(axis = 1)

    df['average_forecasted_returns_of_momentums'] = average_forecasted_returns_of_momentums

    df['short_forecasted_returns_of_momentums']  = short_forecasted_returns_of_momentums 

    df['long_forecasted_returns_of_momentums']  = long_forecasted_returns_of_momentums

    compressed_cols = [
        "average_forecasted_returns_of_momentums",
        "short_forecasted_returns_of_momentums",
        "long_forecasted_returns_of_momentums",
        # "frac_positive_forecast_momentum",
    ]

    sector_column_values = df['Sector'].tolist()

    df = pd.get_dummies(data = df, columns = ['Sector'])
    
    new_columns = [i for i in df.columns if i not in temp_feat_df.columns and i not in compressed_cols]
    for sector_column in new_columns:
        for momentum in compressed_cols:
            df[f'{sector_column}*{momentum}'] = df[sector_column ]*df[momentum]

    df = df.drop(new_columns, axis = 1)
    
    df['Sector'] = sector_column_values

    return df

def popolute_momentum_data(price_data: pd.DataFrame, available_AR_values_per_stock: dict,
                          LOOKBACKS: list,
                          selected_tickers: list):
    
    for ticker in selected_tickers:
        print(ticker)
        p = price_data[f"{ticker}_Close"]
        for lookback in LOOKBACKS:
            print(lookback)
            fill_missing_correlation_coefficients(available_AR_values_per_stock , p, ticker, lookback, 21)  
            
def create_compressed_forecast_features(temp_feat_df):
    df = temp_feat_df.copy()
    
    forecast_features = [i for i in df.columns if 'forecast_return' in i]
    short_term_forecast_features = [i for i in forecast_features if '_5' in i or '_5' in i or '_10' in i or '_25' in i]
    long_term_forecast_features = [i for i in forecast_features if '_60' in i or '_120' in i or '_250 ' in i]
    # scaler = StandardScaler()
    
    # forecast_features_data = scaler.fit_transform(temp_feat_df[forecast_features])
    average_forecasted_returns_of_momentums = df[forecast_features].mean(axis = 1)

    short_forecasted_returns_of_momentums = df[short_term_forecast_features].mean(axis = 1)

    long_forecasted_returns_of_momentums = df[long_term_forecast_features].mean(axis = 1)

    df['average_forecasted_returns_of_momentums'] = average_forecasted_returns_of_momentums

    df['short_forecasted_returns_of_momentums']  = short_forecasted_returns_of_momentums 

    df['long_forecasted_returns_of_momentums']  = long_forecasted_returns_of_momentums

    compressed_cols = [
        "average_forecasted_returns_of_momentums",
        "short_forecasted_returns_of_momentums",
        "long_forecasted_returns_of_momentums",
        # "frac_positive_forecast_momentum",
    ]

    sector_column_values = df['Sector'].tolist()

    df = pd.get_dummies(data = df, columns = ['Sector'])
    
    new_columns = [i for i in df.columns if i not in temp_feat_df.columns and i not in compressed_cols]
    for sector_column in new_columns:
        for momentum in compressed_cols:
            df[f'{sector_column}*{momentum}'] = df[sector_column ]*df[momentum]

    df = df.drop(new_columns, axis = 1)
    
    df['Sector'] = sector_column_values

    return df
