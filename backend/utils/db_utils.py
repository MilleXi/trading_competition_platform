from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class StockData(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Integer, nullable=False)
    ma5 = db.Column(db.Float, nullable=True)
    ma10 = db.Column(db.Float, nullable=True)
    ma20 = db.Column(db.Float, nullable=True)
    rsi = db.Column(db.Float, nullable=True)
    macd = db.Column(db.Float, nullable=True)
    vwap = db.Column(db.Float, nullable=True)
    sma = db.Column(db.Float, nullable=True)
    std_dev = db.Column(db.Float, nullable=True)
    upper_band = db.Column(db.Float, nullable=True)
    lower_band = db.Column(db.Float, nullable=True)
    atr = db.Column(db.Float, nullable=True)
    sharpe_ratio = db.Column(db.Float, nullable=True)
    beta = db.Column(db.Float, nullable=True)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    game_id = db.Column(db.String(36), nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)  # "buy" or "sell"
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)



class GameInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    game_id = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    cash = db.Column(db.Float, nullable=False)  # 当前剩余的现金
    portfolio_value = db.Column(db.Float, nullable=False)  # 股票的总价值
    total_assets = db.Column(db.Float, nullable=False)  # 总资产（现金 + 股票）
    stocks = db.Column(db.JSON, nullable=False)  # 持有的股票数量，以字典形式存储
    score = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

    def update_assets(self):
        self.total_assets = self.cash + self.portfolio_value
        self.last_updated = datetime.utcnow()
        db.session.commit()



class TradeLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    balance = db.Column(db.Float, nullable=False)
    earning = db.Column(db.Float, nullable=False)
    portfolio = db.Column(db.PickleType, nullable=False)
    change = db.Column(db.PickleType, nullable=False)
    earnings_per_stock = db.Column(db.PickleType, nullable=False)
    model = db.Column(db.String(50), nullable=False)
    game_id = db.Column(db.Integer, nullable=False)


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
