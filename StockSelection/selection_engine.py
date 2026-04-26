import ast
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SelectionConfig:
    horizon_bd: int = 21
    train_sample: int | None = 1_000_000
    random_state: int = 42
    model_type: str = "elastic_net"
    train_data_density: str = "soft"
    k_top: int = 8
    corr_thresh: float | None = 0.9
    nan_corr: str = "reject"
    exclude_tickers: tuple[str, ...] = ("sp500",)


def _build_estimator(model_type: str, random_state: int) -> Any:
    if model_type == "linear":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=0.01)
    if model_type == "lasso":
        return Lasso(alpha=0.005, random_state=random_state)
    return ElasticNet(alpha=0.005 / 1.5, l1_ratio=0.5, random_state=random_state)


def _directional_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    if len(yt) == 0 or len(yp) == 0:
        return np.nan
    mask = (yt != 0) & (yp != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask])))


def select_topk_rows_with_corr_filter(
    test_rows: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    k: int = 8,
    corr_thresh: float | None = 0.9,
    nan_corr: str = "reject",
    exclude_tickers: Iterable[str] = (),
    positive_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    if test_rows.empty:
        return pd.DataFrame(columns=test_rows.columns), []

    candidates = test_rows.copy()
    if positive_only and "y_pred" in candidates.columns:
        candidates = candidates[candidates["y_pred"] > 0]
    candidates = candidates.sort_values("y_pred", ascending=False).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    selected_tickers: list[str] = []
    excluded = set(exclude_tickers)

    for _, row in candidates.iterrows():
        ticker = row["Ticker"]
        if ticker in excluded:
            continue

        accept = True
        if corr_thresh is not None and corr_thresh < 1.0:
            ticker_col = f"{ticker}_Close"
            if ticker_col not in price_df.columns:
                accept = nan_corr == "allow"
            else:
                for selected in selected_tickers:
                    selected_col = f"{selected}_Close"
                    if selected_col not in price_df.columns:
                        if nan_corr == "reject":
                            accept = False
                            break
                        continue
                    corr = np.corrcoef(price_df[ticker_col].values, price_df[selected_col].values)[0, 1]
                    if np.isnan(corr):
                        if nan_corr == "reject":
                            accept = False
                            break
                    elif abs(corr) > corr_thresh:
                        accept = False
                        break

        if accept:
            selected_rows.append(row)
            selected_tickers.append(ticker)
            if len(selected_rows) >= k:
                break

    return pd.DataFrame(selected_rows), selected_tickers


def run_experiment_single_day(
    df_all: pd.DataFrame,
    price_df: pd.DataFrame,
    test_date: str | pd.Timestamp,
    *,
    config: SelectionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SelectionConfig()
    df = df_all.dropna(subset=["y"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    test_date = pd.to_datetime(test_date).normalize()
    train_end = test_date - pd.tseries.offsets.BDay(cfg.horizon_bd)

    train_df = df[df["Date"] < train_end].copy()
    if cfg.train_data_density != "dense":
        train_df = train_df[train_df.groupby("Ticker").cumcount() % 21 == 0].copy()
    if cfg.train_sample is not None and len(train_df) > cfg.train_sample:
        train_df = train_df.sample(cfg.train_sample, random_state=cfg.random_state)

    test_df = df[df["Date"] == test_date].copy()

    if train_df.empty or test_df.empty:
        return {
            "TEST_DATE": test_date.date(),
            "Selected tickers (rows mode)": [],
            "Returns": [0.0],
            "Predicted_returns": [],
            "TopK mean realized return": 0.0,
            "Avg_predicted_return": 0.0,
            "Num_stock": 0,
        }

    feature_frame = pd.concat([train_df, test_df], axis=0).sort_values("Date").reset_index(drop=True)
    train_mask = feature_frame["Date"] < train_end
    test_mask = feature_frame["Date"] == test_date

    cat_features = [c for c in ["Sector", "Industry"] if c in feature_frame.columns]
    num_features = (
        feature_frame.drop(columns=["Date", "Ticker", "y"], errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    model_input = pd.get_dummies(feature_frame[cat_features + num_features + ["y"]], columns=cat_features)

    x_train = model_input.loc[train_mask].drop(columns=["y"])
    y_train = model_input.loc[train_mask, "y"]
    x_test = model_input.loc[test_mask].drop(columns=["y"])
    y_test = model_input.loc[test_mask, "y"]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns, index=x_test.index)
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    model = _build_estimator(cfg.model_type, cfg.random_state)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    train_price_df = price_df.copy()
    if not isinstance(train_price_df.index, pd.DatetimeIndex):
        train_price_df.index = pd.to_datetime(train_price_df.index)
    train_price_df = train_price_df[train_price_df.index < train_end]

    test_preds = feature_frame.loc[test_mask, ["Date", "Ticker", "y"]].copy()
    test_preds["y_pred"] = pred_test
    top_rows, _ = select_topk_rows_with_corr_filter(
        test_preds,
        train_price_df,
        k=cfg.k_top,
        corr_thresh=cfg.corr_thresh,
        nan_corr=cfg.nan_corr,
        exclude_tickers=cfg.exclude_tickers,
        positive_only=True,
    )

    if top_rows.empty:
        selected_tickers: list[str] = []
        returns: list[float] = [0.0]
        predicted_returns: list[float] = []
        topk_return = 0.0
        topk_pred = 0.0
    else:
        selected_tickers = top_rows["Ticker"].tolist()
        returns = top_rows["y"].tolist()
        predicted_returns = top_rows["y_pred"].tolist()
        topk_return = float(np.mean(returns))
        topk_pred = float(np.mean(predicted_returns))

    return {
        "TEST_DATE": test_date.date(),
        "HORIZON_BD": int(cfg.horizon_bd),
        "Train end (exclusive)": pd.to_datetime(train_end).date(),
        "Train samples": int(len(x_train)),
        "Test samples (rows)": int(len(x_test)),
        "Train R^2": r2_score(y_train, pred_train) if len(y_train) else np.nan,
        "Train RMSE": np.sqrt(mean_squared_error(y_train, pred_train)) if len(y_train) else np.nan,
        "Train MAE": mean_absolute_error(y_train, pred_train) if len(y_train) else np.nan,
        "Train Directional Acc": _directional_accuracy(y_train, pred_train),
        "Test R^2": r2_score(y_test, pred_test) if len(y_test) else np.nan,
        "Test RMSE": np.sqrt(mean_squared_error(y_test, pred_test)) if len(y_test) else np.nan,
        "Test MAE": mean_absolute_error(y_test, pred_test) if len(y_test) else np.nan,
        "Test Directional Acc": _directional_accuracy(y_test, pred_test),
        "Test Correlation": np.corrcoef(y_test, pred_test)[0, 1] if len(y_test) > 1 else np.nan,
        "Selected tickers (rows mode)": selected_tickers,
        "Returns": returns,
        "Predicted_returns": predicted_returns,
        "TopK mean realized return": topk_return,
        "Avg_predicted_return": topk_pred,
        "Picked rows count": int(len(selected_tickers)),
        "Num_stock": int(len(selected_tickers)),
    }


def run_rolling_backtest_selection(
    df_all: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    config: SelectionConfig | None = None,
    spacing_days: int = 11,
    lookback_windows: int = 8,
    test_dates: Sequence[str | pd.Timestamp] | None = None,
    newest_first: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.
    Prefer `MetaLearner.Backtest.run_selection_backtest` for orchestration.
    """
    from MetaLearner.Backtest import run_selection_backtest

    cfg = config or SelectionConfig()
    return run_selection_backtest(
        df_all=df_all,
        price_df=price_df,
        config=cfg,
        spacing_days=spacing_days,
        lookback_windows=lookback_windows,
        test_dates=test_dates,
        newest_first=newest_first,
        single_day_runner=run_experiment_single_day,
    )


def identify_stocks_for_deployment(
    df_all: pd.DataFrame,
    deploy_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    config: SelectionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SelectionConfig()
    train_df = df_all.dropna(subset=["y"]).copy()
    deploy_df = deploy_df.copy()
    if train_df.empty or deploy_df.empty:
        return {"Selected tickers (rows mode)": [], "Avg_predicted_return": 0.0, "Top Predicted Returns": []}

    train_df["Date"] = pd.to_datetime(train_df["Date"])
    deploy_df["Date"] = pd.to_datetime(deploy_df["Date"])

    cat_features = [c for c in ["Sector", "Industry"] if c in train_df.columns and c in deploy_df.columns]
    num_features = (
        train_df.drop(columns=["Date", "Ticker", "y"], errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.intersection(
            deploy_df.drop(columns=["Date", "Ticker"], errors="ignore")
            .select_dtypes(include=[np.number])
            .columns
        )
        .tolist()
    )

    x_train = pd.get_dummies(train_df[cat_features + num_features], columns=cat_features)
    x_deploy = pd.get_dummies(deploy_df[cat_features + num_features], columns=cat_features)
    x_train, x_deploy = x_train.align(x_deploy, join="left", axis=1, fill_value=0)
    y_train = train_df["y"]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_deploy = pd.DataFrame(imputer.transform(x_deploy), columns=x_deploy.columns, index=x_deploy.index)
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_deploy = pd.DataFrame(scaler.transform(x_deploy), columns=x_deploy.columns, index=x_deploy.index)

    model = _build_estimator(cfg.model_type, cfg.random_state)
    model.fit(x_train, y_train)
    deploy_pred = model.predict(x_deploy)
    deploy_rows = deploy_df.loc[:, ["Date", "Ticker"]].copy()
    deploy_rows["y_pred"] = deploy_pred

    train_price_df = price_df.copy()
    if not isinstance(train_price_df.index, pd.DatetimeIndex):
        train_price_df.index = pd.to_datetime(train_price_df.index)
    train_price_df = train_price_df[train_price_df.index < deploy_df["Date"].max()]

    top_rows, tickers = select_topk_rows_with_corr_filter(
        deploy_rows,
        train_price_df,
        k=cfg.k_top,
        corr_thresh=cfg.corr_thresh,
        nan_corr=cfg.nan_corr,
        exclude_tickers=cfg.exclude_tickers,
        positive_only=True,
    )
    return {
        "TEST_DATE": deploy_df["Date"].max().date(),
        "Selected tickers (rows mode)": tickers,
        "Top Predicted Returns": top_rows["y_pred"].tolist() if not top_rows.empty else [],
        "Avg_predicted_return": float(top_rows["y_pred"].mean()) if not top_rows.empty else 0.0,
    }


def safe_parse_selected_tickers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                return []
        if text:
            return [text]
    return []
