import ccxt
import datetime
import csv


def get_binance_history(symbol, timeframe='1d', limit=365):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = [[datetime.datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d'), candle[1], candle[2], candle[3],
             candle[4], candle[5]] for candle in ohlcv]
    return headers, data


if __name__ == "__main__":
    symbol = 'BTC/USDT'
    headers, data = get_binance_history(symbol)

    output_file = "bitcoin_history.csv"

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)
        csv_writer.writerows(data)

    print(f"Data has been successfully saved to {output_file}.")