import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Blueprint, jsonify, current_app
from backend.utils.db_utils import db, StockData
import yfinance as yf
import pandas as pd


stock_bp = Blueprint('stock', __name__)

STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'C', 'WFC', 'GS',
          'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'XOM', 'CVX', 'COP', 'SLB', 'BKR',
          'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX', 'CAT', 'DE', 'MMM', 'GE', 'HON']

START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
DATA_DIR = 'data'


def calculate_indicators(df):
  df['MA5'] = df['Close'].rolling(window=5).mean()
  df['MA10'] = df['Close'].rolling(window=10).mean()
  df['MA20'] = df['Close'].rolling(window=20).mean()

  delta = df['Close'].diff(1)
  gain = (delta.where(delta > 0, 0)).fillna(0)
  loss = (-delta.where(delta < 0, 0)).fillna(0)
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()
  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))

  df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
  df['SMA'] = df['Close'].rolling(window=50).mean()
  df['Std_dev'] = df['Close'].rolling(window=20).std()

  df['Upper_band'] = df['SMA'] + (df['Std_dev'] * 2)
  df['Lower_band'] = df['SMA'] - (df['Std_dev'] * 2)

  df['ATR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)

  df['Sharpe_Ratio'] = df['Close'].pct_change().rolling(window=252).mean() / df['Close'].pct_change().rolling(
    window=252).std()

  df['Beta'] = df['Close'].pct_change().rolling(window=252).cov(df['Close'].pct_change().rolling(window=252).mean()) / \
               df['Close'].pct_change().rolling(window=252).var()

  df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()

  return df


def download_stock_data():
  with current_app.app_context():
    for stock in STOCKS:
      df = yf.download(stock, start=START_DATE, end=END_DATE)
      # print(df)
      df.reset_index(inplace=True)
      df = calculate_indicators(df)
      for _, row in df.iterrows():
        stock_data = StockData(
          symbol=stock,
          date=row['Date'].date(),
          open=row['Open'],
          high=row['High'],
          low=row['Low'],
          close=row['Close'],
          volume=row['Volume'],
          ma5=row['MA5'],
          ma10=row['MA10'],
          ma20=row['MA20'],
          rsi=row['RSI'],
          macd=row['MACD'],
          vwap=row['VWAP'],
          sma=row['SMA'],
          std_dev=row['Std_dev'],
          upper_band=row['Upper_band'],
          lower_band=row['Lower_band'],
          atr=row['ATR'],
          sharpe_ratio=row['Sharpe_Ratio'],
          beta=row['Beta']
        )
        db.session.add(stock_data)
    db.session.commit()


@stock_bp.route('/stock_data', methods=['GET'])
def get_stock_data():
  stock_data = download_stock_data()
  return jsonify(stock_data)


@stock_bp.route('/stored_stock_data', methods=['GET'])
def get_stored_stock_data():
  with open(os.path.join(DATA_DIR, 'stock_data.pkl'), 'rb') as f:
    stock_data = pd.read_pickle(f)
  return jsonify(stock_data)
