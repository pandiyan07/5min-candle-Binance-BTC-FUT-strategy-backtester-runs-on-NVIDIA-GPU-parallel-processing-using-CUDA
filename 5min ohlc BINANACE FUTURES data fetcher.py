from binance.um_futures import UMFutures
import pandas as pd
import time

def REST_API_DATFET():
    # Initialize the Binance USD-M Futures client
    umfut_rest_api_client = UMFutures()

    # Download the last 40 one-minute candles for BTCUSDT
    candles = umfut_rest_api_client.klines("BTCUSDT", "5m", limit=40)

    # Convert the candle data to a DataFrame
    columns = [
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", 
        "ignore"
    ]
    df = pd.DataFrame(candles, columns=columns)
    df = df[[ "open", "high", "low", "close","close_time"]]

    # Convert timestamp columns to a readable datetime format
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.strftime('%H:%M:%S')

    return df

i=1
while True:
    df = REST_API_DATFET()
    print(df)
    last_close_value = float(df['close'].iloc[-1])
    print('\nData fetched ',i,' time')
    print('the new close value is = ',last_close_value,'\n')
    i+=1
    time.sleep(300)