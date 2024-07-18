import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import os

# 股票列表
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',]  # 科技
tickers += ['JPM', 'BAC', 'C', 'WFC', 'GS',]            # 金融
tickers += ['JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',]         # 医药
tickers += ['XOM', 'CVX', 'COP', 'SLB', 'BKR',]          # 能源
tickers += ['DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',]      # 消费
tickers += ['CAT', 'DE', 'MMM', 'GE', 'HON']            # 工业


num_features_to_keep = 9
start_date = '2020-01-01'
end_date = '2022-01-01'


def get_stock_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    # 日期
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    close = data['Close'].shift(1)

    # 移动平均线
    data['MA5'] = close.rolling(window=5).mean()
    data['MA10'] = close.rolling(window=10).mean()
    data['MA20'] = close.rolling(window=20).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    data.loc[data.index >= RSI.index[0], 'RSI'] = RSI

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    MACD = exp1 - exp2
    data.loc[data.index >= MACD.index[0], 'MACD'] = MACD

    # VWAP
    data['VWAP'] = (close * data['Volume']).cumsum() / data['Volume'].cumsum()

    # Bollinger Bands
    period = 20
    data['SMA'] = close.rolling(window=period).mean()
    data['Std_dev'] = close.rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']

    # 相对大盘的表现
    benchmark_data = yf.download('SPY', start=start_date, end=end_date)['Close']
    data['Relative_Performance'] = (close / benchmark_data.values) * 100

    # 价格变化率
    data['ROC'] = (close.pct_change(periods=1)) * 100

    # 平均变化率
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'].shift(1) - close.shift(1))
    low_close_range = abs(data['Low'].shift(1) - close.shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    data['Close_yes'] = data['Close'].shift(1)
    data['Open_yes'] = data['Open'].shift(1)
    data['High_yes'] = data['High'].shift(1)
    data['Low_yes'] = data['Low'].shift(1)

    data = data.dropna()
    return data


# 获取所有股票数据
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}


def format_feature(data):
    features = ['Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD'\
                , 'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',\
                'Close_yes', 'Open_yes', 'High_yes', 'Low_yes']
    features = []
    X = data[features]
    y = data['Close'].pct_change()
    # y = (data['Close'] - data['Open'])
    X = X.iloc[1:]
    y = y.iloc[1:]
    return X, y


# 格式化数据
stock_features = {ticker: format_feature(data) for ticker, data in stock_data.items()}
# print("stock_features:", stock_features)


def feature_selection_for_stocks(stock_features, best_params, num_features_to_keep=8):
    feature_importances_all = {}
    stock_features_selected = {}

    # 训练模型，获得所有股票的特征重要度
    for ticker, (X, y) in stock_features.items():
        params = best_params

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **params)
        xgb_model.fit(X, y)
        feature_importances = xgb_model.feature_importances_

        feature_importance_list = [(feature, importance) for feature, importance in zip(X.columns, feature_importances)]
        feature_importance_list.sort(key=lambda x: x[1], reverse=True)

        feature_importances_all[ticker] = feature_importance_list

    # 打印每支股票的特征重要度排名
    for ticker, importance_list in feature_importances_all.items():
        # print(f"Stock: {ticker}")
        for feature, importance in importance_list:
            pass
            # print(f"Importance of {feature}: {importance:.2f}")
    # print(feature_importances_all)
    feature_importance_totals = {}
    # 平均值
    for stock, feature_importances in feature_importances_all.items():
        for feature, importance in feature_importances:
            if feature in feature_importance_totals:
                feature_importance_totals[feature] += importance
            else:
                feature_importance_totals[feature] = importance
    num_stocks = len(feature_importances_all)
    average_feature_importances = {feature: importance / num_stocks for feature, importance in
                                   feature_importance_totals.items()}
    # average_feature_importances = sorted(average_feature_importances, key=lambda x: x[1], reverse=True)
    for feature, avg_importance in average_feature_importances.items():
        print(f"Average importance of {feature} across all stocks: {avg_importance:.2f}")

    # 只保留重要的特征
    importance_list = sorted(average_feature_importances.items(), key=lambda x: x[1], reverse=True)
    importance_list = importance_list[:num_features_to_keep]
    important_features = [feature for feature, importance in importance_list[:num_features_to_keep]]
    print(f"Selected the top{num_features_to_keep} stocks: {important_features}")
    for stock in stock_features:
        # X_selected = stock_features[stock][0]
        stock_features_selected[stock] = stock_features[stock]
    return stock_features_selected


# 进行特征选择
temp_params = {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 50, 'subsample': 0.6}
# stock_features_selected = feature_selection_for_stocks(stock_features, temp_params, num_features_to_keep)
stock_features_selected = stock_features




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda')

scaler_y = MinMaxScaler()
scaler_X = MinMaxScaler()


def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_and_predict_lstm(ticker, data, X, y, n_steps=60, num_epochs=500, batch_size=32, learning_rate=0.001):
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps-1:-1]

    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_test = X_train[split_index-n_steps+1:]

    y_test = y_train[split_index-n_steps+1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1+split_index, len(X_scaled)+1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]),
                                   dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(y.index[i-1])
            # print(f"Date: {y.index[i]}, Predicted: {(1 + y_pred[0][0]) * data['Close'].iloc[i - 1]:.2f}, Actual: {data['Close'].iloc[i]:.2f}")
            predict_percentages.append(y_pred[0][0]*100)
            actual_percentages.append(y[i-1]*100)
    # predict_percentages = [(pred / X['Close'].iloc[i - 1] - 1) * 100 for i, pred in enumerate(predictions, n_steps)]
    # actual_percentages = y[split_index+n_steps:].dropna() * 100

    delta = [p - a for p, a in zip(predict_percentages, actual_percentages)]
    result = pd.DataFrame({
        'predict_percentages(%)': predict_percentages,
        'actual_percentages(%)': actual_percentages,
        'delta(%)': delta
    })
    print(result)

    # 简单策略的收益率
    print("Naive strategy earn rate:", sum(actual_percentages), "%")

    # LSTM策略的收益率
    lstm_strategy_earn = []
    for i, predict in enumerate(predict_percentages):
        if predict > 0:
            # print("earn:", actual_percentages[i])
            lstm_strategy_earn.append(actual_percentages[i])
        else:
            lstm_strategy_earn.append(0)
    print("LSTM strategy earn rate:", sum(lstm_strategy_earn), "%")

    acc=0
    for pred1, pred2 in zip(predict_percentages, actual_percentages):
        print("pred:",pred1, "actu:",pred2)
        if pred1 * pred2 > 0:
            acc+=1
    acc/=len(predict_percentages)
    print("acc:",acc)


    # 绘制累积收益率曲线
    cumulative_naive_percentage = np.cumsum(actual_percentages)
    cumulative_lstm_percentage = np.cumsum(lstm_strategy_earn)
    plt.figure(figsize=(10, 6))
    plt.plot(test_indices, cumulative_naive_percentage, marker='o', markersize=3, linestyle='-', color='blue',
             label='Naive Strategy')
    plt.plot(test_indices, cumulative_lstm_percentage, marker='o', markersize=3, linestyle='-', color='orange',
             label='LSTM Strategy')
    plt.title(f'Daily Earnings Percentages for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if not os.path.exists('pic'):
        os.makedirs('pic')
    plt.savefig(f'pic/{ticker}.png')
    plt.close()

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result


all_predictions_lstm = {}
for ticker in tickers:
    data = stock_data[ticker]
    X, y = stock_features_selected[ticker]
    predict_result = train_and_predict_lstm(ticker, data, X, y)
    all_predictions_lstm[ticker] = predict_result
all_predictions_lstm







start = (int((len(data) - 1) * 0.8)+1)
start_test_date = y.iloc[start:start+1].index[0]
end = (len(data) - 2)
end_test_date = y.iloc[end:end+1].index[0]
# start_test_date = start_test_date.to_pydatetime()
# end_test_date = end_test_date.to_pydatetime()
print(end_test_date)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math


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
        self.data = self.download_data()
        self.strategy = "Naive"

    def download_data(self):
        symbols = self.symbols + ['SPY']  # download the benchmark data
        data = yf.download(symbols, start=self.start_date, end=self.end_date + timedelta(days=30))['Close']
        return data

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
        plt.show()

        if not os.path.exists('pic'):
            os.makedirs('pic')
        plt.savefig(f'pic/{self.strategy}_backtest.png')
        plt.close()

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

        # buy the selected stocks using all the cash
        num_stocks = len(selected_symbols)
        use_cash = 0
        for symbol in selected_symbols:
            self.portfolio[symbol] = math.floor((self.cash / num_stocks) / self.data[symbol][current_date])
            use_cash += self.data[symbol][current_date] * self.portfolio[symbol]
        self.cash -= use_cash


naive_strategy = NaiveStrategy(symbols=tickers, start_date=start_test_date, end_date=end_test_date)
naive_strategy.plot_portfolio_value()


class ShortStrategy(MyStrategy):
    def __init__(self, symbols, start_date, end_date, model_predict):
        super().__init__(symbols, start_date, end_date, model_predict)
        self.strategy = "LSTM"

    def rebalance(self, current_date):
        print(current_date)
        for symbol, shares in self.portfolio.items():
            print("price:", self.data[symbol][current_date], "share:", shares)

        portfolio_value = sum(self.data[symbol][current_date] * shares for symbol, shares in self.portfolio.items())
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

        # set the weights for selected symbols
        weights = {}
        for symbol, model_predict in self.model_predict.items():
            model_predict_tomorrow = model_predict[str(tomorrow)]
            if not symbol in selected_symbols:
                continue
            weights[symbol] = model_predict_tomorrow

        weight_sum = sum(weights.values())
        weights = {key: value / weight_sum for key, value in weights.items()}

        # buy the selected stocks using all the cash
        position = self.adjust_position(current_date)
        num_stocks = len(selected_symbols)
        use_cash = 0
        for symbol in selected_symbols:
            symbol_weight = weights[symbol]
            self.portfolio[symbol] = math.floor(
                (self.cash * position * symbol_weight) / self.data[symbol][current_date])
            use_cash += self.data[symbol][current_date] * self.portfolio[symbol]
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


lstm_strategy = ShortStrategy(symbols=tickers, start_date=start_test_date, end_date=end_test_date,
                             model_predict=all_predictions_lstm)
lstm_strategy.plot_portfolio_value()


for record in lstm_strategy.trade_log:
    print(f"Date: {record['date']}")
    print(f"Balance: {record['balance']:.2f}")
    print(f"Earning: {record['earning']:.2f}")
    print("Portfolio:")
    for symbol, shares in record['portfolio'].items():
        print(f"  {symbol}: {shares} shares")
    print("Change:")
    for symbol, shares in record['change'].items():
        print(f"  {symbol}: {shares} shares")
    print("Earnings_per_stock:")
    for symbol, earn in record['earnings_per_stock'].items():
        print(f"  {symbol}: {earn}")
    print("-" * 40)


import pickle
if not os.path.exists('prediction/'):
  os.makedirs('prediction')
if not os.path.exists('../backend/predictions/LSTM'):
  os.makedirs('../backend/predictions/LSTM')
for ticker, predictions in all_predictions_lstm.items():
  file_path = os.path.join('../backend/predictions/LSTM', f'{ticker}_predictions.pkl')
  with open(file_path, 'wb') as file:
    pickle.dump(predictions, file)
    print(f'Saved predictions for {ticker} to {file_path}')
