from flask import Blueprint, request, jsonify
from utils.db_utils import db, TradeLog
from datetime import datetime
import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3000"}})


strategy_bp = Blueprint('strategy_bp', __name__)


class MyStrategy:
  def __init__(self, symbols, start_date, end_date, model_predict={}):
    self.symbols = symbols
    self.start_date = start_date
    self.end_date = end_date
    self.model_predict = model_predict
    self.trade_log = []
    self.Initialize()

  def Initialize(self):
    self.cash = 100000
    self.risk_free_rate = 0
    self.rebalance_time = 1
    self.last_rebalance = self.start_date
    self.portfolio = {}
    self.prev_price = {}
    self.data, self.data_open = self.download_data()
    self.strategy = "Naive"

  def download_data(self):
    symbols = self.symbols + ['SPY']  # download the benchmark data
    data = yf.download(symbols, start=self.start_date, end=self.end_date + timedelta(days=30))['Close']
    data_open = yf.download(symbols, start=self.start_date, end=self.end_date + timedelta(days=30))['Open']
    return data, data_open

  def calculate_sharpe_ratio(self, returns):
    excess_returns = returns - self.risk_free_rate / 252
    std = excess_returns.std()
    return np.sqrt(252) * (excess_returns.mean() / std) if std != 0 else 0

  def calculate_beta(self, stock_returns, benchmark_returns):
    assert len(stock_returns) == len(benchmark_returns)
    covariance_matrix = np.cov(stock_returns, benchmark_returns)
    return covariance_matrix[0, 1] / np.var(benchmark_returns)

  def is_bull_market(self, current_date):
    recent_market_data = self.data['SPY'][
      (self.data.index <= current_date) & (self.data.index > current_date - timedelta(days=3))]
    recent_market_returns = recent_market_data.pct_change().dropna()
    return recent_market_returns.mean() > 0

  def adjust_position(self, current_date):
    recent_market_data = self.data['SPY'][
      (self.data.index <= current_date) & (self.data.index > current_date - timedelta(days=3))]
    recent_market_returns = recent_market_data.pct_change().dropna()
    market_mean_return = recent_market_returns.mean()
    if math.isnan(market_mean_return):
      return 1.0
    elif market_mean_return > 0:
      return 1.0
    elif market_mean_return > -0.02:
      return 0.8
    else:
      # print(market_mean_return)
      return max(0.8 - (market_mean_return + 0.02) * 10, 0)

  def rebalance(self, current_date):
    pass

  def run_backtest(self):
    # rebalance_dates = pd.date_range(self.start_date, self.end_date, freq=f'{self.rebalance_time}D')
    trading_days = self.data.index
    rebalance_dates = trading_days[::self.rebalance_time]
    portfolio_values = []
    # print(rebalance_dates)
    for date in self.data.index:
      if date in rebalance_dates and date != rebalance_dates[-1]:
        # print("before:", self.portfolio)
        # print(date)
        self.rebalance(date)
        # print("after:", self.portfolio)
      # print("cash:", self.cash)
      # print("date:", date)
      portfolio_value = self.cash + sum(
        self.data[symbol][date] * shares for symbol, shares in self.portfolio.items())
      # print(portfolio_value)
      portfolio_values.append(portfolio_value)

    return portfolio_values

  def plot_portfolio_value(self):
    portfolio_values = self.run_backtest()
    portfolio_series = pd.Series(portfolio_values, index=self.data.index)
    portfolio_series.plot(title="Portfolio Value Over Time", figsize=(10, 6))
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    if not os.path.exists('pic'):
      os.makedirs('pic')
    if not os.path.exists('pic/strategy'):
      os.makedirs('pic/strategy')
    plt.savefig(f'pic/strategy/{self.strategy}_backtest.png')
    plt.close()
    plt.show()

    print("Initial funding:", 100000)
    print("Final funding:", portfolio_series.iloc[-1])
    print("Yield rate:", (portfolio_series.iloc[-1] - 100000) / 100000 * 100, "%")


class NaiveStrategy(MyStrategy):
  def __init__(self, symbols, start_date, end_date):
    super().__init__(symbols, start_date, end_date)

  def rebalance(self, current_date):
    portfolio_value = sum(self.data[symbol][current_date] * shares for symbol, shares in self.portfolio.items())
    self.cash += portfolio_value
    self.portfolio = {}

    selected_symbols = self.symbols

    num_stocks = len(selected_symbols)
    use_cash = 0
    for symbol in selected_symbols:
      self.portfolio[symbol] = math.floor((self.cash / num_stocks) / self.data[symbol][current_date])
      use_cash += self.data[symbol][current_date] * self.portfolio[symbol]
    self.cash -= use_cash


class ShortStrategy(MyStrategy):
  def __init__(self, symbols, start_date, end_date, model_predict):
    super().__init__(symbols, start_date, end_date, model_predict)
    self.strategy = "LSTM"


  def rebalance(self, current_date):
    print(current_date)
    for symbol, shares in self.portfolio.items():
      print("price:", self.data[symbol][current_date], "share:", shares)

    portfolio_value = sum(self.data_open[symbol][current_date] * shares for symbol, shares in self.portfolio.items())
    self.cash += portfolio_value

    # Clear current portfolio
    previous_portfolio = self.portfolio.copy()
    self.portfolio = {}

    # calculate the sharpe ratio and the beta value of every stock
    sharpes = {}
    betas = {}
    for symbol in self.symbols:
      hist = self.data[symbol][
        (self.data.index <= current_date) & (self.data.index > current_date - timedelta(days=365))]
      returns = hist.pct_change().dropna()
      sharpes[symbol] = self.calculate_sharpe_ratio(returns)

      # 计算beta值
      benchmark_hist = self.data['SPY'][
        (self.data.index <= current_date) & (self.data.index > current_date - timedelta(days=365))]
      benchmark_returns = benchmark_hist.pct_change().dropna()
      betas[symbol] = self.calculate_beta(returns, benchmark_returns)
    print("sharpes:", sharpes)

    weights = {}
    selected_symbols = []
    current_date_index = self.data.index.get_loc(current_date)
    tomorrow = self.data.index[current_date_index + 1]
    # print("tomorrow:", tomorrow)
    for symbol, model_predict in self.model_predict.items():
      # print(model_predict)
      if (tomorrow > self.end_date):
        return
      model_predict_tomorrow = model_predict[str(tomorrow)]
      # print("date:", current_date, "symbol:", symbol, "predict:", model_predict_tomorrow)
      if model_predict_tomorrow > 0:
        selected_symbols.append(symbol)

    for symbol in selected_symbols:
      if sharpes[symbol] < -0.5:
        selected_symbols.remove(symbol)

    # Buy the selected stocks using all the cash with stop loss
    for symbol in self.symbols:
      # Check stop loss condition using previous portfolio
      if self.prev_price:
        prev_price = self.prev_price[symbol]
        current_price = self.data[symbol][current_date]
        if (current_price - prev_price) / prev_price < -0.1:  # The stop loss is set to 10%
          if symbol in selected_symbols:
            selected_symbols.remove(symbol)  # Skip buying this stock

    plus_symbols = []
    if self.is_bull_market(current_date):
      plus_symbols = [symbol for symbol in self.symbols if betas.get(symbol, 0) > 1]
    else:
      plus_symbols = [symbol for symbol in self.symbols if betas.get(symbol, 0) < 0.2]
    for symbol in plus_symbols:
      if symbol not in selected_symbols:
        selected_symbols.append(symbol)


    # set the weights for selected symbols
    weights = {}
    for symbol, model_predict in self.model_predict.items():
      model_predict_tomorrow = model_predict[str(tomorrow)]
      if not symbol in selected_symbols:
        continue
      weights[symbol] = model_predict_tomorrow

    for symbol in plus_symbols:
      weights[symbol] = 0.3


    weight_sum = sum(weights.values())
    weights = {key: value / weight_sum for key, value in weights.items()}

    # for symbol in self.symbols:
    #   if symbol in weights.keys():
    #     weights[symbol] += sharpes[symbol]*3
    #     if weights[symbol] < 0:
    #       weights[symbol] = 0
    #
    # weight_sum = sum(weights.values())
    # weights = {key: value / weight_sum for key, value in weights.items()}


    # buy the selected stocks using all the cash
    position = self.adjust_position(current_date)
    # for symbol, sharpe in sharpes.items():
    #   print("Sharpe11111:", sharpe)
    #   position += (sharpe*5)
    # position *= 0.5
    # if position > 1:
    #   position = 1
    # print("Position11111:", position)

    num_stocks = len(selected_symbols)
    use_cash = 0
    for symbol in selected_symbols:
      symbol_weight = weights[symbol]
      self.portfolio[symbol] = math.floor(
        (self.cash * position * symbol_weight) / self.data_open[symbol][current_date])
      use_cash += self.data_open[symbol][current_date] * self.portfolio[symbol]
    self.cash -= use_cash

    position_change = {}
    for symbol in self.symbols:
      previous_shares = previous_portfolio.get(symbol, 0)
      current_shares = self.portfolio.get(symbol, 0)
      change = current_shares - previous_shares
      if change != 0:
        position_change[symbol] = change

    if not self.trade_log:
      earnings_per_stock = {}
    else:
      earnings_per_stock = self.trade_log[-1]['earnings_per_stock'].copy()
      # print(earnings_per_stock)
    for symbol in self.symbols:
      if symbol in previous_portfolio:
        # print(previous_portfolio, symbol)
        # print(self.prev_price)
        prev_price = self.prev_price[symbol]
        curr_price = self.data[symbol][current_date]
        hold = previous_portfolio[symbol]
        if symbol not in earnings_per_stock.keys():
          # print(1)
          # print(symbol)
          earnings_per_stock[symbol] = (curr_price - prev_price) * hold
        else:
          # print(2)
          earnings_per_stock[symbol] += (curr_price - prev_price) * hold
    # print("after:", earnings_per_stock)

    self.prev_price = {}

    for symbol in self.symbols:
      self.prev_price[symbol] = self.data[symbol][current_date]
      # print(self.prev_price[symbol])

    # record the log
    portfolio_value = sum(self.data[symbol][current_date] * shares for symbol, shares in self.portfolio.items())
    if self.trade_log:
      previous_balance = self.trade_log[-1]['balance']
    else:
      previous_balance = 100000
    trade_record = {
      'date': current_date,
      'balance': portfolio_value + self.cash,
      'earning': portfolio_value + self.cash - previous_balance,
      'portfolio': self.portfolio.copy(),
      'change': position_change,
      'earnings_per_stock': earnings_per_stock,
    }
    self.trade_log.append(trade_record)
    # print(self.trade_log)


@strategy_bp.route('/run_strategy', methods=['POST'])
@cross_origin()
def run_strategy():
  data = request.json
  tickers = data['tickers']
  game_id = data['game_id']

  start_test_date = data['start_test_date']
  start_test_date_str = data['start_test_date']
  start_test_date = datetime.strptime(start_test_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
  # print("data:", data)
  # print("tikers:", tickers)
  # print("game_id:", game_id)
  # start_test_date = data['start_test_date']
  print("start_test_date111111111111:", start_test_date)
  # start_test_date = datetime(2023, 1, 1)
  end_test_date = datetime(2024, 1, 1)
  print("end_test_date111111111111:", end_test_date)

  directory_path = 'predictions/LSTM'
  all_predictions = {}
  for ticker in tickers:
    file_path = os.path.join(directory_path, f'{ticker}_predictions.pkl')
    try:
      with open(file_path, 'rb') as file:
        all_predictions[ticker] = pickle.load(file)
        print(f'Successfully loaded {file_path}')
    except FileNotFoundError:
      print(f'File not found: {file_path}')
      return jsonify({"error": f"File not found: {file_path}"}), 404
  # print("all_predictions")
  lstm_strategy = ShortStrategy(symbols=tickers, start_date=start_test_date, end_date=end_test_date,
                                model_predict=all_predictions)
  # print(1111122222)
  lstm_strategy.run_backtest()
  # print(1111133333)

  trade_log = lstm_strategy.trade_log
  # print("trade_log:", trade_log)
  return jsonify({"trade_log": trade_log})


@strategy_bp.route('/save_trade_log', methods=['POST'])
@cross_origin()
def save_trade_log():
  data = request.json
  date_str = data['date']
  try:
    date = datetime.strptime(date_str, '%Y-%m-%d')
  except ValueError:
    try:
      # Handle the case where date is in a different format
      date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
      date = datetime.strptime(date, '%Y-%m-%d')
    except Exception as e:
      return jsonify({"error": f"Date format error: {e}"}), 400

  balance = data['balance']
  earning = data['earning']
  portfolio = data['portfolio']
  change = data['change']
  earnings_per_stock = data['earnings_per_stock']
  model = data['model']
  game_id = data['game_id']

  new_record = TradeLog(
    date=date,
    balance=balance,
    earning=earning,
    portfolio=portfolio,
    change=change,
    earnings_per_stock=earnings_per_stock,
    model=model,
    game_id=game_id
  )
  db.session.add(new_record)
  db.session.commit()
  return jsonify({"message": "Trade log saved successfully."})


@strategy_bp.route('/get_trade_log', methods=['GET'])
@cross_origin()
def get_trade_log():
    game_id = request.args.get('game_id')
    model = request.args.get('model')
    date_str = request.args.get('date')
    print("game_id", game_id)
    print("model", model)
    print("date_str", date_str)
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        # print("date:", date)
    except ValueError:
        return jsonify({"error": "Date format error"}), 400

    trade_log = TradeLog.query.filter_by(game_id=game_id, model=model, date=date).first()
    # print("trade_log:", trade_log)
    if not trade_log:
        return jsonify({"error": "Trade log not found"}), 404

    return jsonify({
        "date": trade_log.date.strftime('%Y-%m-%d'),
        "balance": trade_log.balance,
        "earning": trade_log.earning,
        "portfolio": trade_log.portfolio,
        "change": trade_log.change,
        "earnings_per_stock": trade_log.earnings_per_stock,
        "model": trade_log.model,
        "game_id": trade_log.game_id
    })
