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
    user_id = db.Column(db.Integer, nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)  # "buy" or "sell"
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)


class GameInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    game_id = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    balance = db.Column(db.Float, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)


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
