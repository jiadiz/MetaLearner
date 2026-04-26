
def build_original_mean_reversion_features(price_data, price_column, sector_etf_column, sp500_column):
       # Horizons (trading days)
    import numpy as np
    import pandas as pd
    # D1  = 21       # ~1 month
    # D3  = 63       # ~3 months
    D6  = 126      # ~6 months
    D12 = 252      # ~12 months
    D24 = 504

    p = price_data[price_column].copy()
    s = price_data[sector_etf_column].copy()
    sp = price_data[sp500_column].copy()

    p_r =  np.log(p / p.shift(1)).copy()
    s_r =  np.log(s / s.shift(1)).copy()
    sp_r =  np.log(sp / sp.shift(1)).copy()

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

    def get_beta(x, y):
        x = np.array(x).reshape(-1, 1)

        lm = LinearRegression()
        lm.fit(x, y)
        beta = lm.coef_[0]
        return beta


    df = pd.DataFrame({'Stock_price':p,
                       'ETF_price':s,
                       'SP500_price':sp,
        'Stock_return':p_r,
                       'Sector_return':s_r,
                       'Econ_return':sp_r,
                       # 'sector_tracking_error_d1':sector_tracking_error_d1,
                       # 'sector_tracking_error_d1':sector_tracking_error_d3,
                       'vr_5d':vr_5d,
                       'vr_20d':vr_20d,
                       'vr_60d':vr_60d,
                       'vr_120d':vr_120d,

                       'sector_tracking_error_d1':sector_tracking_error_d6,
                       'sector_tracking_error_d1':sector_tracking_error_d12,
                       'sector_tracking_error_d1':sector_tracking_error_d24,
                       # 'econ_tracking_error_d1':econ_tracking_error_d1,
                       # 'econ_tracking_error_d1':econ_tracking_error_d3,
                       'econ_tracking_error_d1':econ_tracking_error_d6,
                       'econ_tracking_error_d1':econ_tracking_error_d12,
                       'econ_tracking_error_d1':econ_tracking_error_d24,
                      })

    from statsmodels.tsa.stattools import adfuller
    import numpy as np
    import pandas as pd

    # def rolling_regression(x, y, window):
    #     n = len(x)

    #     betas      = np.full(n, np.nan)
    #     resid_std  = np.full(n, np.nan)
    #     z_scores   = np.full(n, np.nan)
    #     adf_pvals  = np.full(n, np.nan)

    #     lm = LinearRegression()

    #     for i in range(window, n):
    #         X_win = x.iloc[i-window:i].values.reshape(-1, 1)
    #         y_win = y.iloc[i-window:i].values

    #         # fit regression
    #         lm.fit(X_win, y_win)
    #         beta = lm.coef_[0]
    #         betas[i] = beta

    #         # residuals in the window
    #         y_pred_win = lm.predict(X_win)
    #         residuals = y_win - y_pred_win

    #         # std of residuals (sample std)
    #         std = residuals.std(ddof=1)
    #         resid_std[i] = std

    #         # z-score of *today's* residual
    #         x_today = np.array([[x.iloc[i]]])
    #         y_pred_today = lm.predict(x_today)[0]
    #         resid_today = y.iloc[i] - y_pred_today
    #         z_scores[i] = resid_today / std if std != 0 else np.nan

    #         # ADF p-value on window residuals
    #         try:
    #             adf_res = adfuller(residuals, autolag='AIC')
    #             adf_pvals[i] = adf_res[1]   # p-value
    #         except Exception:
    #             adf_pvals[i] = np.nan       # too short / numerical issue

    #     return betas, resid_std, z_scores, adf_pvals 
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.stattools import adfuller

    def rolling_regression(x, y, window):
        """
        x, y: pandas Series of (typically) daily returns for benchmark (x) and stock (y)
        window: lookback window for the rolling regression / ADF / z-score etc.
        """

        n = len(x)

        betas      = np.full(n, np.nan)
        resid_std  = np.full(n, np.nan)
        z_scores   = np.full(n, np.nan)
        adf_pvals  = np.full(n, np.nan)

        # --- extra residual-based features ---
        if window >= 252:
            res_mom_12_1 = np.full(n, np.nan)
        else:
            res_mom_06_1  = np.full(n, np.nan)

        res_mom_3m   = np.full(n, np.nan)
        res_rev_5d   = np.full(n, np.nan)
        res_rsi_14   = np.full(n, np.nan)
        res_streak   = np.full(n, np.nan)

        macd_line    = np.full(n, np.nan)
        macd_signal  = np.full(n, np.nan)
        macd_hist    = np.full(n, np.nan)

    # store the one-day residuals series (epsilon_t at each t)
        resid_series = np.full(n, np.nan)

        lm = LinearRegression()

    # --- hyperparameters for residual features ---
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

    # EMA state for MACD
        ema_fast = np.full(n, np.nan)
        ema_slow = np.full(n, np.nan)
        ema_sig  = np.full(n, np.nan)

    # --- helper functions on residual subseries ---

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

    # --- main rolling loop ---

        for i in range(window, n):

            X_win = x.iloc[i-window:i]

            print(X_win.index[-1], X_win.index[0])
            X_win = X_win.values.reshape(-1, 1)
            y_win = y.iloc[i-window:i].values

            # fit regression
            lm.fit(X_win, y_win)
            beta = lm.coef_[0]
            betas[i] = beta

            # residuals in the window (for std & ADF)
            y_pred_win = lm.predict(X_win)
            residuals_window = y_win - y_pred_win

            # std of residuals (sample std)
            std = residuals_window.std(ddof=1)
            resid_std[i] = std

            # today's residual (one-step ahead) using today's x,y
            x_today = np.array([[x.iloc[i]]])
            y_pred_today = lm.predict(x_today)[0]
            resid_today = y.iloc[i] - y_pred_today
            resid_series[i] = resid_today

            # z-score of today's residual
            z_scores[i] = resid_today / std if std != 0 else np.nan

            # ADF p-value on window residuals
            try:
                adf_res = adfuller(residuals_window, autolag='AIC')
                adf_pvals[i] = adf_res[1]   # p-value
            except Exception:
                adf_pvals[i] = np.nan       # too short / numerical issue

            # ---- residual-based momentum / reversal / RSI / streak ----
            resid_hist = resid_series[:i+1]

            # 12–1 residual momentum (~12m excluding last 1m)
            if window >= 252:
                res_mom_12_1[i] = residual_momentum(
                    resid_hist,
                    length=MOM_12_LEN,
                    skip=MOM_12_SKIP
                    )
            else:
                res_mom_06_1[i] = residual_momentum(
                    resid_hist,
                    length=MOM_06_LEN,
                    skip=MOM_12_SKIP
                    )


            # 3-month residual momentum
            res_mom_3m[i] = residual_momentum(
            resid_hist,
            length=MOM_3M_LEN,
            skip=0
            )

            # 5-day residual reversal
            res_rev_5d[i] = residual_reversal(
                resid_hist,
                length=REV_5D_LEN
            )

            # 14-day RSI on residuals
            res_rsi_14[i] = residual_rsi(
                resid_hist,
                length=RSI_LEN
            )

            # positive residual streak length
            res_streak[i] = residual_streak_length(resid_hist)

            # ---- MACD on residuals (EMA-based, updated incrementally) ----
            ema_fast[i] = update_ema(ema_fast[i-1], resid_today, alpha_fast)
            ema_slow[i] = update_ema(ema_slow[i-1], resid_today, alpha_slow)

            if not (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
                macd_line[i] = ema_fast[i] - ema_slow[i]
            else:
                macd_line[i] = np.nan

            # signal line EMA on MACD
            ema_sig[i] = update_ema(ema_sig[i-1], macd_line[i], alpha_sig)
            macd_signal[i] = ema_sig[i]

            if not (np.isnan(macd_line[i]) or np.isnan(macd_signal[i])):
                macd_hist[i] = macd_line[i] - macd_signal[i]
            else:
                macd_hist[i] = np.nan


        return (
            betas,
            resid_std,
            z_scores,
            adf_pvals,
            res_mom_12_1 if window >= 252 else res_mom_06_1,
            res_mom_3m,
            res_rev_5d,
            res_rsi_14,
            res_streak,
            macd_line,
            macd_signal,
            macd_hist,
        )


    for D in [
            # D1, D3,
                # D6,
        # D12, D24
    ]:
        print(D, ' period computed')
        info = rolling_regression(
            df['Sector_return'].dropna(),
            df['Stock_return'].dropna(),
            #            df['Stock_price'].dropna(),
            # df['ETF_price'].dropna(),
        window=D
        )

        beta, std, z, adf_p, res_mom_long_1, res_mom_3m, res_rev_5d, res_rsi_14, res_streak, macd_line, macd_signal, macd_hist = info


        df[f'sector_r_beta_d{D}'] = beta
        df[f'sector_r_resid_std_d{D}'] = std
        df[f'sector_r_resid_z_d{D}'] = z
        df[f'sector_r_resid_adf_p_d{D}'] = adf_p
        df[f'sector_r_z_times_1_minus_p_value_{D}'] = z * (1 - adf_p)
        if D >= 252:
            df[f'sector_r_res_mom_12_1_{D}'] = res_mom_long_1
        else:
            df[f'sector_r_res_mom_06_1_{D}'] = res_mom_long_1
        df[f'sector_r_res_rev_5d_{D}'] = res_rev_5d
        df[f'sector_r_res_rsi_14_{D}'] = res_rsi_14
        df[f'sector_r_res_streak_{D}'] = res_streak
        df[f'sector_r_res_rsi_14_{D}'] = res_rsi_14
        df[f'sector_r_macd_line_{D}'] = macd_line
        df[f'sector_r_macd_signal_{D}'] = macd_signal
        df[f'sector_r_macd_hist_{D}'] = macd_hist

    for D in [
        # D1, D3, 
        # D6, 
        # D12, D24
        ]:
        print(D, ' period computed')
        info  = rolling_regression(
            df['Econ_return'],
            df['Stock_return'],
            #            df['Stock_price'].dropna(),
            # df['sp500_price'].dropna(),
            window=D
        )
        beta, std, z, adf_p, res_mom_long_1, res_mom_3m, res_rev_5d, res_rsi_14, res_streak, macd_line, macd_signal, macd_hist = info

        df[f'econ_r_beta_d{D}'] = beta
        df[f'econ_r_resid_std_d{D}'] = std
        df[f'econ_r_resid_z_d{D}'] = z
        df[f'econ_r_resid_adf_p_d{D}'] = adf_p
        df[f'econ_r_z_times_1_minus_p_value_{D}'] = z * (1 - adf_p)
        if D >= 252:
            df[f'econ_res_mom_12_1_{D}'] = res_mom_long_1
        else:
            df[f'econ_res_mom_06_1_{D}'] = res_mom_long_1
        df[f'econ_r_res_rev_5d_{D}'] = res_rev_5d
        df[f'econ_r_res_rsi_14_{D}'] = res_rsi_14
        df[f'econ_r_res_streak_{D}'] = res_streak
        df[f'econ_r_res_rsi_14_{D}'] = res_rsi_14
        df[f'econ_r_macd_line_{D}'] = macd_line
        df[f'econ_r_macd_signal_{D}'] = macd_signal
        df[f'econ_r_macd_hist_{D}'] = macd_hist

    stock_price = df['Stock_price']

    for D in [
        # D1,
        # D3, 
        # D6,
        # D12, D24
    ]:
        print(D, ' period computed')
        info  = rolling_regression(
            df['SP500_price'],
            df['Stock_price'],

            #            df['Stock_price'].dropna(),
            # df['sp500_price'].dropna(),
            window=D
        )
        beta, std, z, adf_p, res_mom_long_1, res_mom_3m, res_rev_5d, res_rsi_14, res_streak, macd_line, macd_signal, macd_hist = info

        df[f'econ_p_beta_d{D}'] = beta
        df[f'econ_p_resid_std_d{D}'] = std/stock_price 
        df[f'econ_p_resid_z_d{D}'] = z
        df[f'econ_p_resid_total_deviation_per_price_d{D}'] = (std*z)/stock_price 
        df[f'econ_p_resid_adf_p_d{D}'] = adf_p
        df[f'econ_p_z_times_1_minus_p_value_{D}'] = z * (1 - adf_p)
        df[f'econ_p_resid_std_times_z_times_1_minus_p_value_{D}'] = (std * z * (1 - adf_p))/ stock_price 
        if D >= 252:
            df[f'econ_p_res_mom_12_1_{D}'] = res_mom_long_1/stock_price
        else:
            df[f'econ_p_res_mom_06_1_{D}'] = res_mom_long_1/stock_price
        df[f'econ_p_res_rev_5d_{D}'] = res_rev_5d/stock_price
        df[f'econ_p_res_rsi_14_{D}'] = (res_rsi_14*std*z*adf_p)/stock_price
        df[f'econ_p_res_streak_{D}'] = (res_streak*std*z*adf_p)/stock_price
        df[f'econ_p_macd_line_{D}'] = macd_line/stock_price
        df[f'econ_p_macd_signal_{D}'] = macd_signal/stock_price
        df[f'econ_p_macd_hist_{D}'] = macd_hist/stock_price

    for D in [
        # D1,
        # D3, 
        D6,
        # D12, D24
    ]:
        print(D, ' period computed')
        info  = rolling_regression(
            df['ETF_price'],
            df['Stock_price'],
            #            df['Stock_price'].dropna(),
            # df['sp500_price'].dropna(),
            window=D
        )
        print(df['ETF_price'].shape)
        beta, std, z, adf_p, res_mom_long_1, res_mom_3m, res_rev_5d, res_rsi_14, res_streak, macd_line, macd_signal, macd_hist = info

        df[f'sector_p_beta_d{D}'] = beta
        df[f'sector_p_resid_std_d{D}'] = std/stock_price 
        df[f'sector_p_resid_z_d{D}'] = z
        df[f'sector_p_resid_total_deviation_per_price_d{D}'] = (std*z)/stock_price 
        df[f'sector_p_resid_adf_p_d{D}'] = adf_p
        df[f'sector_p_z_times_1_minus_p_value_{D}'] = z * (1 - adf_p)
        df[f'sector_p_resid_std_times_z_times_1_minus_p_value_{D}'] = (std * z * (1 - adf_p))/ stock_price 
        if D >= 252:
            df[f'sector_p_res_mom_12_1_{D}'] = res_mom_long_1/stock_price
        else:
            df[f'sector_p_res_mom_06_1_{D}'] = res_mom_long_1/stock_price
        df[f'sector_p_res_rev_5d_{D}'] = res_rev_5d/stock_price
        df[f'sector_p_res_rsi_14_{D}'] = (res_rsi_14*std*z*adf_p)/stock_price
        df[f'sector_p_res_streak_{D}'] = (res_streak*std*z*adf_p)/stock_price
        df[f'sector_p_macd_line_{D}'] = macd_line/stock_price
        df[f'sector_p_macd_signal_{D}'] = macd_signal/stock_price
        df[f'sector_p_macd_hist_{D}'] = macd_hist/stock_price
    return df
