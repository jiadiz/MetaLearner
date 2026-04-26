
import pandas as pd


def filter_for_date(
    price_data: pd.DataFrame,
    start_date: str = '2020-01-01'
    ) -> pd.DataFrame:

    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data = price_data[price_data['Date'] >= pd.to_datetime(start_date)]
    return price_data

def drop_columns_with_na(price_data: pd.DataFrame)-> pd.DataFrame:

    before_dropping_na_tickers = [i for i in price_data.columns if 'close' in i.lower()]
    close_na_values = price_data.isna().sum()

    price_data = price_data.dropna(axis = 1)
    after_dropping_na_tickers = [i for i in price_data.columns if 'close' in i.lower()]

    dropped_columns = [i for i in before_dropping_na_tickers if i not in after_dropping_na_tickers]
    dropped_columns_na_counts = close_na_values[close_na_values.index.isin(dropped_columns)]
    print('Dropped columns', dropped_columns_na_counts.index.tolist())
    print('Their missing values are,', dropped_columns_na_counts.sort_values())
    return price_data

def create_column_of_days_after_2024(price_data: pd.DataFrame)-> pd.DataFrame:
    dates = pd.to_datetime(price_data['Date'])
    dates = pd.to_datetime(dates, utc=True)

    dates_series = pd.to_datetime(dates)

    # Convert all datetimes to UTC to ensure consistency
    dates_series_utc = dates_series.dt.tz_convert('UTC')

    # Define the reference date in UTC (or any specific timezone)
    reference_date = dates.min()

    # Calculate the difference in days and create a new DataFrame
    days_after_2024 = (dates_series_utc - reference_date).dt.days

    # Handle negative values (before 2024) if desired
    # days_after_2024 = days_after_2024.apply(lambda x: x if x >= 0 else float('nan'))
    days_after_2024.index = price_data.index

    price_data['days'] = days_after_2024

    return price_data

def create_week_number_column(price_data: pd.DataFrame ) -> pd.DataFrame:
    price_data ['Date'] = pd.to_datetime(price_data ['Date'])

    # Set the start date
    start_date = pd.Timestamp('2008-01-01')
    # Create a week 0 for any dates before the first Monday of 2008
    first_monday = start_date + pd.offsets.Week(weekday=0)

    # Define a function to assign week numbers
    def get_week_number(date):
        if date < first_monday:
            return 0  # Assign week 0 for dates before the first Monday of 2008
        else:
            return ((date - first_monday).days // 7) + 1

    # Apply the function to create a new 'week' column
    price_data ['week'] = price_data ['Date'].apply(get_week_number)
    return price_data 

