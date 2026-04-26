"""
Create daily price series CSV from S&P 500, S&P MidCap 400, and sector ETFs.
Hardcoded ticker lists - no Wikipedia or external fetches.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path


# S&P 500 constituents (as of notebook snapshot)
SP500_TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL",
    "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AMTM", "AEE", "AEP", "AXP",
    "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON",
    "APA", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ",
    "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL",
    "BAC", "BAX", "BDX", "BRK.B", "BBY", "TECH", "BIIB", "BLK", "BX", "BK",
    "BA", "BKNG", "BWA", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR",
    "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX",
    "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "COR", "CNC",
    "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI",
    "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH",
    "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW",
    "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI",
    "CVS", "DHR", "DRI", "DVA", "DAY", "DECK", "DE", "DELL", "DAL", "DVN",
    "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW",
    "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW",
    "EA", "ELV", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX",
    "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD",
    "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
    "FSLR", "FE", "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN",
    "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS",
    "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL", "HIG", "HAS",
    "HCA", "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON",
    "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
    "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG",
    "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J",
    "JNJ", "JCI", "JPM", "JNPR", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB",
    "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW",
    "LVS", "LDOS", "LEN", "LLY", "LIN", "LYV", "LKQ", "LMT", "L", "LOW",
    "LULU", "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA",
    "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM",
    "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR",
    "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM",
    "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH",
    "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON",
    "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PARA", "PH",
    "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW",
    "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG",
    "PTC", "PSA", "PHM", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX",
    "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP",
    "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW",
    "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX",
    "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW",
    "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN",
    "TPL", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC",
    "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI",
    "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VTRS",
    "VICI", "V", "VST", "VMC", "WRB", "GWW", "WAB", "WBA", "WMT", "DIS",
    "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WY", "WMB",
    "WTW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
]

# S&P MidCap 400 constituents (as of notebook snapshot)
SP_MIDCAP_400_TICKERS = [
    "AA", "AAL", "AAON", "ACI", "ACM", "ADC", "AEIS", "AFG", "AGCO", "AHR",
    "AIT", "ALGM", "ALK", "ALLY", "ALV", "AM", "AMG", "AMH", "AMKR", "AN",
    "ANF", "APG", "APPF", "AR", "ARMK", "ARW", "ASB", "ASGN", "ASH", "ATI",
    "ATR", "AVAV", "AVNT", "AVT", "AVTR", "AXTA", "AYI", "BAH", "BBWI", "BC",
    "BCO", "BDC", "BHF", "BILL", "BIO", "BJ", "BKH", "BLD", "BLKB", "BMRN",
    "BRBR", "BRKR", "BROS", "BRX", "BSY", "BURL", "BWA", "BWXT", "BYD",
    "CACI", "CAR", "CART", "CASY", "CAVA", "CBSH", "CBT", "CCK", "CDP",
    "CELH", "CFR", "CG", "CGNX", "CHDN", "CHE", "CHH", "CHRD", "CHWY", "CIEN",
    "CLF", "CLH", "CMC", "CNH", "CNM", "CNO", "CNX", "CNXC", "COHR", "COKE",
    "COLB", "COLM", "COTY", "CPRI", "CR", "CRBG", "CROX", "CRS", "CRUS",
    "CSL", "CUBE", "CUZ", "CVLT", "CW", "CXT", "CYTK", "DAR", "DBX", "DCI",
    "DINO", "DKS", "DLB", "DOCS", "DOCU", "DT", "DTM", "DUOL", "DY", "EEFT",
    "EGP", "EHC", "ELAN", "ELF", "ELS", "ENS", "ENSG", "ENTG", "EPR", "EQH",
    "ESAB", "ESNT", "EVR", "EWBC", "EXEL", "EXLS", "EXP", "EXPO", "FAF",
    "FBIN", "FCFS", "FCN", "FFIN", "FHI", "FHN", "FIVE", "FLEX", "FLG", "FLO",
    "FLR", "FLS", "FN", "FNB", "FND", "FNF", "FOUR", "FR", "FTI", "G", "GAP",
    "GATX", "GBCI", "GEF", "GGG", "GHC", "GLPI", "GME", "GMED", "GNTX", "GPK",
    "GT", "GTLS", "GTM", "GWRE", "GXO", "H", "HAE", "HALO", "HGV", "HIMS",
    "HL", "HLI", "HLNE", "HOG", "HOMB", "HQY", "HR", "HRB", "HWC", "HXL",
    "IBOC", "IDA", "ILMN", "INGR", "IPGP", "IRT", "ITT", "JAZZ", "JEF", "JHG",
    "JLL", "KBH", "KBR", "KD", "KEX", "KMPR", "KNF", "KNSL", "KNX", "KRC",
    "KRG", "KTOS", "LAD", "LAMR", "LEA", "LECO", "LFUS", "LITE", "LIVN",
    "LNTH", "LOPE", "LPX", "LSCC", "LSTR", "M", "MANH", "MASI", "MAT", "MEDP",
    "MIDD", "MKSI", "MLI", "MMS", "MORN", "MP", "MSA", "MSM", "MTDR", "MTG",
    "MTN", "MTSI", "MTZ", "MUR", "MUSA", "MZTI", "NBIX", "NEU", "NFG", "NJR",
    "NLY", "NNN", "NOV", "NOVT", "NSA", "NTNX", "NVST", "NVT", "NWE", "NXST",
    "NXT", "NYT", "OC", "OGE", "OGS", "OHI", "OKTA", "OLED", "OLLI", "OLN",
    "ONB", "ONTO", "OPCH", "ORA", "ORI", "OSK", "OVV", "OZK", "PAG", "PATH",
    "PB", "PBF", "PCTY", "PEGA", "PEN", "PFGC", "PII", "PINS", "PK", "PLNT",
    "PNFP", "POR", "POST", "PPC", "PR", "PRI", "PSN", "PSTG", "PVH", "QLYS",
    "R", "RBA", "RBC", "REXR", "RGA", "RGEN", "RGLD", "RH", "RLI", "RMBS",
    "RNR", "ROIV", "RPM", "RRC", "RRX", "RS", "RYAN", "RYN", "SAIA", "SAIC",
    "SAM", "SARO", "SATS", "SBRA", "SCI", "SEIC", "SF", "SFM", "SGI", "SHC",
    "SIGI", "SLAB", "SLGN", "SLM", "SMG", "SNX", "SON", "SPXC", "SR", "SSB",
    "SSD", "ST", "STAG", "STRL", "STWD", "SWX", "SYNA", "TCBI", "TEX", "THC",
    "THG", "THO", "TKR", "TLN", "TMHC", "TNL", "TOL", "TREX", "TRU", "TTC",
    "TTEK", "TTMI", "TWLO", "TXNM", "TXRH", "UBSI", "UFPI", "UGI", "ULS",
    "UMBF", "UNM", "USFD", "UTHR", "VAL", "VC", "VFC", "VLY", "VMI", "VNO",
    "VNOM", "VNT", "VOYA", "VVV", "WAL", "WBS", "WCC", "WEX", "WFRD", "WH",
    "WHR", "WING", "WLK", "WMG", "WMS", "WPC", "WSO", "WTFC", "WTRG", "WTS",
    "WWD", "XPO", "XRAY", "YETI", "ZION",
]

# Sector ETFs + SPY
ETF_TICKERS = [
    "XLK", "XLF", "XLI", "XLV", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLE",
    "XLB", "SPY",
]

SP500_TICKER = ['^GSPC']

def download_daily_data(ticker: str, start: str = "2018-01-01") -> pd.DataFrame | None:
    """Download daily OHLCV data for a ticker from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, interval="1d", progress=False, threads=False)
        if data.empty:
            return None
        return data
    except Exception:
        return None


def create_price_series_csv(
    output_path: str | Path = "all_daily_adjusted_close.csv",
    start_date: str = "2018-01-01",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download price data for S&P 500, S&P MidCap 400, and sector ETFs,
    combine into a single DataFrame, and save to CSV.

    Parameters
    ----------
    output_path : str or Path
        Path for the output CSV file.
    start_date : str
        Start date for historical data (YYYY-MM-DD).
    verbose : bool
        If True, print progress for each ticker.

    Returns
    -------
    pd.DataFrame
        The combined price data (Volume + Close per ticker).
    """
    stock_tickers = SP500_TICKERS + SP_MIDCAP_400_TICKERS
    tickers =  stock_tickers[:10] + ETF_TICKERS + SP500_TICKER

    data = {}
    for ticker in tickers:
        if verbose:
            print(f"Downloading data for {ticker}...")
        df = download_daily_data(ticker, start=start_date)
        if df is not None and not df.empty:
            data[ticker] = df
        elif verbose:
            print(f"  Could not download data for {ticker}")

    close_data = {}
    for ticker, df in data.items():
        stock_data = df[["Volume", "Close"]].copy()
        if ticker == '^GSPC':
            ticker = 'sp500'
        stock_data.columns = [f"{ticker}_Volume", f"{ticker}_Close"]
        close_data[ticker] = stock_data

    if not close_data:
        raise ValueError("No data was downloaded for any ticker.")

    # pd.concat with dict uses keys as column level; droplevel(0) yields ticker-prefixed names
    train_data = pd.concat(close_data, axis=1)
    train_data = train_data.reset_index()
    train_data.columns = train_data.columns.droplevel(0)
    train_data.columns.values[0] = "Date"

    output_path = Path(output_path)
    train_data.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved {len(train_data)} rows to {output_path}")

    return train_data


if __name__ == "__main__":
    create_price_series_csv()
