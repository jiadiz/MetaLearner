
import random
from sklearn.metrics import r2_score
import pandas as pd
import yfinance as yf

def download_daily_data(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="max", interval="1d")
    return data

# Example list

# Randomly select 10 points
def find_tickers(combined_data: pd.DataFrame) -> pd.DataFrame:
    existing_tickers = []
    for i in combined_data.columns:
        if 'Close' in i:
            ticker = i.split('_')[0]
            if ticker != 'sp500':
                existing_tickers.append(ticker)
    return existing_tickers



def check_data_quality(combined_data: pd.DataFrame, existing_tickers: list)-> pd.DataFrame:
    testing_tickers = random.sample(existing_tickers, 10)

    first_hand_data = {}
    # Download data for each ticker
    for ticker in testing_tickers:
        print(f"Downloading data for {ticker}...")
        try:
            first_hand_data[ticker] = download_daily_data(ticker)
            # print(data[ticker])
        except Exception as e:
            print(f"Could not download data for {ticker}: {e}")

    for ticker in first_hand_data:
        data = first_hand_data[ticker]
        data = data.reset_index().rename(columns = {'Close': f'{ticker}_Close'})
        data['Date'] = pd.to_datetime(data['Date'])

        adj_close_column = f'{ticker}_Close'
        tested_data = combined_data[['Date', adj_close_column]].rename(columns = {adj_close_column: f'{adj_close_column}_tested'})
        testing_data = data[['Date', adj_close_column]].rename(columns = {adj_close_column: f'{adj_close_column}_testing'})
        testing_data .columns = [i[0] for i in testing_data.columns]
        print(tested_data.columns, testing_data.columns)
        tested_and_testing = pd.merge(tested_data, testing_data, how = 'left', on ='Date')
        tested_price = tested_and_testing[f'{adj_close_column}_tested']
        testing_price = tested_and_testing[f'{adj_close_column}_testing']
        r2 = r2_score(tested_price, testing_price)
        print(r2)
        if r2 < 0.98:
            print('Data quality has issue')
            return
    print('Data is good')
    return
