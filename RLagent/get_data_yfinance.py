import yfinance as yf
import os

# 指定股票代码列表
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'JPM', 'BAC', 'C', 'WFC', 'GS',
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',
    'XOM', 'CVX', 'COP', 'SLB', 'BKR',
    'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',
    'CAT', 'DE', 'MMM', 'GE', 'HON'
]

# 设置时间范围
start_date = "2023-01-01"
end_date = "2024-01-01"

# 创建保存数据的文件夹
folder_name = "yeardata"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 获取数据并保存为CSV
for ticker in tickers:
    # 下载数据
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # 检查是否下载成功
    if data.empty:
        print(f"数据下载失败: {ticker}")
        continue
    
    # 添加年份到文件名
    file_name = f"{folder_name}/{ticker}_2023.csv"
    
    # 保存数据到CSV文件
    data.to_csv(file_name, index_label='Date')
    print(f"已保存: {file_name}")
