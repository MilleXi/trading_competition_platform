import sys
import os

from flask import Blueprint, jsonify, current_app, request
from utils.db_utils import db, StockData
import yfinance as yf
import pandas as pd


stock_bp = Blueprint('stock', __name__)

STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'C', 'WFC', 'GS',
          'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'XOM', 'CVX', 'COP', 'SLB', 'BKR',
          'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX', 'CAT', 'DE', 'MMM', 'GE', 'HON']

START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
DATA_DIR = 'data'


def download_stock_data():
  with current_app.app_context():
    for stock in STOCKS:
      df = yf.download(stock, start=START_DATE, end=END_DATE)
      # print(df)
      df.reset_index(inplace=True)
      for _, row in df.iterrows():
        stock_data = StockData(
          symbol=stock,
          date=row['Date'].date(),
          open=row['Open'],
          high=row['High'],
          low=row['Low'],
          close=row['Close'],
          volume=row['Volume']
        )
        db.session.add(stock_data)
    db.session.commit()


# @stock_bp.route('/stored_stock_data', methods=['GET'])
# def get_stored_stock_data():
#     stocks = StockData.query.all()
#     stock_data = [{
#         'symbol': stock.symbol,
#         'date': stock.date,
#         'open': stock.open,
#         'high': stock.high,
#         'low': stock.low,
#         'close': stock.close,
#         'volume': stock.volume
#     } for stock in stocks]
#     return jsonify(stock_data)


@stock_bp.route('/stored_stock_data', methods=['GET'])
def get_stored_stock_data():
    symbol = request.args.get('symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    query = StockData.query
    if symbol:
        query = query.filter_by(symbol=symbol)
    if start_date:
        query = query.filter(StockData.date >= start_date)
    if end_date:
        query = query.filter(StockData.date <= end_date)

    stocks = query.all()
    stock_data = [{
        'symbol': stock.symbol,
        'date': stock.date,
        'open': stock.open,
        'high': stock.high,
        'low': stock.low,
        'close': stock.close,
        'volume': stock.volume
    } for stock in stocks]
    return jsonify(stock_data)
