"""
First-principles reconstruction of the Fama-French 3 factors (MKT_RF, SMB, HML)
plus RF from *daily* data, aggregated to monthly.

Why "first principles"?
-----------------------
The FF factors are not single-asset returns, so naively compounding the *net*
daily factor (``prod(1 + factor) - 1``) is wrong:

  * ``MKT_RF`` is an excess return = market total return - risk-free. The correct
    monthly value compounds the market leg and the RF leg *separately*, then
    differences them.
  * ``SMB`` / ``HML`` are long-short spreads of the 6 size x book-to-market
    portfolios. The correct monthly value compounds each underlying portfolio to
    monthly, *then* forms the spread.

Compounding the net spread injects a spurious cross-term. Validated against the
official monthly factors (2000-present, 316 months), the first-principles method
reduces mean error from ~10-14 bp/month to ~2 bp for SMB/HML (R^2 0.994 -> 0.9999)
and is marginally better for MKT_RF, while the absolute "sum of daily" method is
materially worse (max errors 150-270 bp).

Data source: Ken French's data library (the same files pandas_datareader wraps),
downloaded directly so this module has no dependency on pandas_datareader (which
is incompatible with pandas 3.x).

Primary entry point: ``reconstruct_fama_factors()``.
"""
from __future__ import annotations

import io
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FACTORS_DAILY_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
PORTFOLIOS_6_DAILY_ZIP = "6_Portfolios_2x3_daily_CSV.zip"

# US Treasury daily par-yield curve CSV (per calendar year). The "1 Mo" column is
# the constant-maturity 1-month T-bill yield (annualized %), available from ~2001.
TREASURY_YIELD_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
TREASURY_1M_START_YEAR = 2001

# Column groupings for the 6 (2x3) size x book-to-market value-weighted portfolios.
_SMALL_PORTFOLIOS = ["SMALL LoBM", "ME1 BM2", "SMALL HiBM"]
_BIG_PORTFOLIOS = ["BIG LoBM", "ME2 BM2", "BIG HiBM"]
_VALUE_PORTFOLIOS = ["SMALL HiBM", "BIG HiBM"]
_GROWTH_PORTFOLIOS = ["SMALL LoBM", "BIG LoBM"]

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/zip,*/*",
}

# French files flag missing observations with these sentinels.
_MISSING_SENTINELS = [-99.99, -999.0]


# ------------------------------- downloading -------------------------------

def _download_french_csv(zip_name: str, timeout: int = 60) -> str:
    """Download a French-library ``*_CSV.zip`` and return its inner CSV as text."""
    resp = requests.get(FRENCH_BASE + zip_name, headers=_BROWSER_HEADERS, timeout=timeout)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    return z.read(z.namelist()[0]).decode("latin-1")


def _parse_french_block(raw: str, date_len: int) -> pd.DataFrame:
    """
    Parse the first contiguous block of date-keyed rows from a French CSV.

    French CSVs start with a descriptive preamble, then a header row like
    ``,SMALL LoBM,ME1 BM2,...`` (or ``,Mkt-RF,SMB,HML,RF``), then rows keyed by a
    ``YYYYMMDD`` (``date_len=8``) or ``YYYYMM`` (``date_len=6``) date. The block
    ends at the first non-date row (blank line / start of the annual or
    equal-weighted section), which is exactly what we want to keep.
    """
    lines = raw.splitlines()
    pat = re.compile(r"^\s*\d{" + str(date_len) + r"}\s*$")
    start = None
    for i, ln in enumerate(lines):
        toks = ln.split(",")
        if toks and pat.match(toks[0]):
            start = i
            break
    if start is None:
        raise ValueError("Could not locate a date-keyed data block in the CSV.")

    cols = [c.strip() for c in lines[start - 1].split(",")]
    cols[0] = "Date"
    rows = []
    for ln in lines[start:]:
        toks = ln.split(",")
        if not (toks and pat.match(toks[0])):
            break
        rows.append([x.strip() for x in toks])

    df = pd.DataFrame(rows, columns=cols).set_index("Date")
    return df.apply(pd.to_numeric, errors="coerce")


def load_ff_factors_daily() -> pd.DataFrame:
    """Daily FF factors as decimals: columns ``MKT_RF, SMB, HML, RF``."""
    df = _parse_french_block(_download_french_csv(FACTORS_DAILY_ZIP), 8)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={"Mkt-RF": "MKT_RF"})
    return df / 100.0


def load_ff_6portfolios_daily() -> pd.DataFrame:
    """
    Daily value-weighted returns (decimals) for the 6 (2x3) size x book-to-market
    portfolios: columns SMALL LoBM, ME1 BM2, SMALL HiBM, BIG LoBM, ME2 BM2, BIG HiBM.
    """
    df = _parse_french_block(_download_french_csv(PORTFOLIOS_6_DAILY_ZIP), 8)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(_MISSING_SENTINELS, np.nan)
    return df / 100.0


def load_treasury_1m_yield(start_year: int, end_year: int, timeout: int = 40) -> pd.Series:
    """
    Daily US Treasury 1-month constant-maturity par yield (annualized %) from
    home.treasury.gov, concatenated across ``start_year..end_year``.

    Returns a Series indexed by date (name ``treasury_1m_yield``). Years that
    fail to download, or that predate the 1-month series (~2001), are skipped;
    if nothing is retrieved an empty Series is returned (callers fall back).
    """
    frames: list[pd.DataFrame] = []
    for yr in range(start_year, end_year + 1):
        url = TREASURY_YIELD_CSV.format(year=yr)
        try:
            resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:  # network / parse issue for this year
            warnings.warn(f"Treasury 1-month yield fetch failed for {yr}: {exc!r}")
            continue
        if "Date" in df.columns and "1 Mo" in df.columns:
            frames.append(df[["Date", "1 Mo"]])

    if not frames:
        return pd.Series(dtype=float, name="treasury_1m_yield")

    t = pd.concat(frames, ignore_index=True)
    t["Date"] = pd.to_datetime(t["Date"], format="%m/%d/%Y", errors="coerce")
    s = pd.to_numeric(t.set_index("Date")["1 Mo"], errors="coerce").dropna().sort_index()
    s.name = "treasury_1m_yield"
    return s


def _treasury_monthly_rf(yield_daily: pd.Series) -> pd.Series:
    """
    Convert the daily 1-month T-bill yield into a monthly risk-free *return*,
    matching Fama-French's beginning-of-month convention: take the first
    observation in each calendar month and compound the annualized yield to a
    one-month holding-period return ``(1 + y)**(1/12) - 1``.

    Returns a Series indexed by ``PeriodIndex('M')``.
    """
    if yield_daily.empty:
        return pd.Series(dtype=float)
    bom = yield_daily.groupby(yield_daily.index.to_period("M")).first()  # beginning-of-month
    return (1.0 + bom / 100.0) ** (1.0 / 12.0) - 1.0


# ----------------------------- reconstruction ------------------------------

def _compound_by_month(daily: pd.Series, periods: pd.PeriodIndex) -> pd.Series:
    """Geometrically compound a daily return series within each calendar month."""
    return (1.0 + daily).groupby(periods).prod() - 1.0


def reconstruct_fama_factors(
    start: str | None = None,
    end: str | None = None,
    factors_daily: pd.DataFrame | None = None,
    portfolios_daily: pd.DataFrame | None = None,
    rf_method: str = "treasury",
    treasury_yields: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Reconstruct monthly Fama-French factors from daily data using first principles.

    Method (validated to ~1-2 bp/month vs the official monthly factors):
      * ``MKT_RF``: rebuild the market total return ``Rm = MKT_RF + RF``, compound
        ``Rm`` and ``RF`` separately within each month, then difference.
      * ``SMB``: compound each of the 6 size/B-M portfolios to monthly, then
        ``mean(small 3) - mean(big 3)``.
      * ``HML``: same, then ``mean(value 2) - mean(growth 2)``.
      * ``RF``: see ``rf_method``.

    Parameters
    ----------
    start, end : str or None
        Optional date bounds (e.g. ``"2015-01-01"``) applied to the *daily* data
        before monthly aggregation. Months only partially covered by the bound
        will be computed from the available days.
    factors_daily, portfolios_daily : pd.DataFrame or None
        Optional pre-loaded daily inputs (e.g. cached) to avoid re-downloading.
        If omitted, they are fetched from the French library.
    rf_method : {"treasury", "compound"}
        How to build the monthly RF column.
          * ``"treasury"`` (default): beginning-of-month US Treasury 1-month
            yield compounded to a 1-month return. This matches the official
            monthly RF to ~1 bp (R^2 ~ 0.993) -- far better than compounding the
            daily RF. Months not covered by the Treasury series (pre-2001, or if
            the download fails) automatically fall back to the compounded-daily
            RF below.
          * ``"compound"``: monthly compound of the daily FF RF. Available for
            the full history but only ~0.91 R^2 vs the official monthly RF
            (trading-day vs calendar-day mismatch + ZIRP rounding).
    treasury_yields : pd.Series or None
        Optional pre-loaded daily Treasury 1-month yield series (see
        ``load_treasury_1m_yield``) to avoid re-downloading. Only used when
        ``rf_method="treasury"``.

    Returns
    -------
    pd.DataFrame
        Indexed by month-end ``DatetimeIndex``, columns
        ``['MKT_RF', 'SMB', 'HML', 'RF']`` in decimal monthly-return units.
    """
    if rf_method not in ("treasury", "compound"):
        raise ValueError(f"rf_method must be 'treasury' or 'compound', got {rf_method!r}")

    fac = load_ff_factors_daily() if factors_daily is None else factors_daily.copy()
    p6 = load_ff_6portfolios_daily() if portfolios_daily is None else portfolios_daily.copy()

    if start is not None:
        fac = fac.loc[fac.index >= pd.Timestamp(start)]
        p6 = p6.loc[p6.index >= pd.Timestamp(start)]
    if end is not None:
        fac = fac.loc[fac.index <= pd.Timestamp(end)]
        p6 = p6.loc[p6.index <= pd.Timestamp(end)]

    per_fac = fac.index.to_period("M")
    per_p6 = p6.index.to_period("M")

    # MKT_RF: compound the market and risk-free legs separately, then difference.
    market_total = fac["MKT_RF"] + fac["RF"]
    mkt_rf = _compound_by_month(market_total, per_fac) - _compound_by_month(fac["RF"], per_fac)

    # RF: compounded-daily RF is the always-available baseline / fallback.
    rf = _compound_by_month(fac["RF"], per_fac)
    if rf_method == "treasury":
        if treasury_yields is None:
            ymin = max(TREASURY_1M_START_YEAR, rf.index.min().year)
            ymax = rf.index.max().year
            treasury_yields = (
                load_treasury_1m_yield(ymin, ymax)
                if ymax >= TREASURY_1M_START_YEAR
                else pd.Series(dtype=float)
            )
        rf_treasury = _treasury_monthly_rf(treasury_yields)
        if rf_treasury.empty:
            warnings.warn(
                "Treasury 1-month yield unavailable; RF falls back to compounded "
                "daily FF RF (lower accuracy)."
            )
        else:
            # Overwrite only the months the Treasury series covers; earlier months
            # keep the compounded-daily fallback so full history is preserved.
            common = rf.index.intersection(rf_treasury.index)
            rf.loc[common] = rf_treasury.loc[common]

    # SMB / HML: compound each underlying portfolio, then form the spread.
    comp_p = {col: _compound_by_month(p6[col], per_p6) for col in p6.columns}
    small = sum(comp_p[c] for c in _SMALL_PORTFOLIOS) / len(_SMALL_PORTFOLIOS)
    big = sum(comp_p[c] for c in _BIG_PORTFOLIOS) / len(_BIG_PORTFOLIOS)
    value = sum(comp_p[c] for c in _VALUE_PORTFOLIOS) / len(_VALUE_PORTFOLIOS)
    growth = sum(comp_p[c] for c in _GROWTH_PORTFOLIOS) / len(_GROWTH_PORTFOLIOS)
    smb = small - big
    hml = value - growth

    out = pd.DataFrame({"MKT_RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf})
    out = out.dropna(how="all")
    out.index = out.index.to_timestamp("M")  # stamp each row at month-end
    out.index.name = "Date"
    return out


if __name__ == "__main__":
    factors = reconstruct_fama_factors(start="2015-01-01")
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print(f"Reconstructed {factors.shape[0]} months x {factors.shape[1]} factors")
    print(factors.tail())
