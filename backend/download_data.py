
import os
import sys
from flask import Flask

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db_utils import db, init_db
from routes.stock_routes import download_stock_data

# 创建临时 Flask 应用以初始化数据库
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
init_db(app)


def main():
    with app.app_context():
        # 确保数据目录存在
        DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(DATA_DIR, exist_ok=True)

        # 调用 download_stock_data 函数下载并存储股票数据
        print("Starting to download stock data...")
        download_stock_data()
        print("Stock data downloaded and saved.")


if __name__ == '__main__':
    main()
