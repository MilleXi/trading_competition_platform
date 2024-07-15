from flask import Flask
from flask_cors import CORS
from routes.stock_routes import stock_bp
from routes.transaction_routes import transaction_bp
from routes.game_routes import game_bp
from utils.db_utils import db, init_db
import os

app = Flask(__name__)
CORS(app)

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
init_db(app)

# 注册蓝图
app.register_blueprint(stock_bp, url_prefix='/api')
app.register_blueprint(transaction_bp, url_prefix='/api')
app.register_blueprint(game_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
