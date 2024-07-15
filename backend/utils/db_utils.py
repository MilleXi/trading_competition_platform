from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from backend.singleton_db import Database

db_instance = Database()
db = db_instance.db


class StockData(db.Model):
    __tablename__ = 'stock_data'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
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
    relative_performance = db.Column(db.Float, nullable=True)
    atr = db.Column(db.Float, nullable=True)
    sharpe_ratio = db.Column(db.Float, nullable=True)
    beta = db.Column(db.Float, nullable=True)


class Transaction(db.Model):
    __tablename__ = 'stock_data'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)  # "buy" or "sell"
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)


class GameInfo(db.Model):
    __tablename__ = 'stock_data'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    balance = db.Column(db.Float, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
