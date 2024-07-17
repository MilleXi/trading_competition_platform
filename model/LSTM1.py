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
tickers = ['PG', 'JPM', 'V', 'UNH', 'DIS']

# 参数
param_grid = {
    'max_depth': [4, 6, 8, 10, 12],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [25, 50, 100, 200]
}

num_features_to_keep = 9


def get_stock_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
    # 日期
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day

    # 移动平均线
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    data.loc[data.index >= RSI.index[0], 'RSI'] = RSI

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    MACD = exp1 - exp2
    data.loc[data.index >= MACD.index[0], 'MACD'] = MACD

    # VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # Bollinger Bands
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']

    # 相对大盘的表现
    benchmark_data = yf.download('SPY', start='2020-01-01', end='2024-01-01')['Close']
    data['Relative_Performance'] = (data['Close'] / benchmark_data.values) * 100

    # 价格变化率
    data['ROC'] = (data['Close'].pct_change(periods=1)) * 100

    # 平均变化率
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    data = data.dropna()
    return data


# 获取所有股票数据
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}

def format_feature(data):
    features = ['Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD'\
                , 'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR']
    X = data[features]
    y = data['Close'].pct_change()
    # y = (data['Close'] - data['Open'])
    X = X.iloc[1:]
    y = y.iloc[1:]
    return X, y

# 格式化数据
stock_features = {ticker: format_feature(data) for ticker, data in stock_data.items()}
print("stock_features:", stock_features)


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
stock_features_selected = feature_selection_for_stocks(stock_features, temp_params, num_features_to_keep)






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


def train_and_predict_lstm(ticker, data, X, y, n_steps=60, num_epochs=100, batch_size=32, learning_rate=0.001):
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
        for i in range(1+split_index, len(X_scaled)):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]),
                                   dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(y.index[i-1])
            print(
                f"Date: {y.index[i]}, Predicted: {(1 + y_pred[0][0]) * data['Close'].iloc[i - 1]:.2f}, Actual: {data['Close'].iloc[i]:.2f}")
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
    plt.savefig(f'pic/{ticker}_lstm.png')

    plt.show()

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result


all_predictions_lstm = {}
for ticker in tickers:
    data = stock_data[ticker]
    X, y = stock_features_selected[ticker]
    predict_result = train_and_predict_lstm(ticker, data, X, y)
    all_predictions_lstm[ticker] = predict_result
all_predictions_lstm
