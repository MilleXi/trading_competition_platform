from flask import Flask
from flask_cors import CORS
from routes.stock_routes import stock_bp
from routes.transaction_routes import transaction_bp
from routes.game_routes import game_bp
from routes.strategy_routes import strategy_bp
from utils.db_utils import db, init_db
import os
import json


def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})

    # 配置数据库
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/db.sqlite3'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 初始化数据库
    init_db(app)

    # 注册蓝图
    app.register_blueprint(stock_bp, url_prefix='/api')
    app.register_blueprint(transaction_bp, url_prefix='/api')
    app.register_blueprint(game_bp, url_prefix='/api')
    app.register_blueprint(strategy_bp, url_prefix='/api')

    # 确保数据目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)

    # 初始化交易记录文件和游戏信息文件
    if not os.path.exists('records/transactions.json'):
        with open('records/transactions.json', 'w') as f:
            json.dump([], f)

    if not os.path.exists('records/game_info.json'):
        with open('records/game_info.json', 'w') as f:
            json.dump({}, f)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000)
