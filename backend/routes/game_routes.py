from flask import Blueprint, request, jsonify
import os
import json
import pandas as pd
from datetime import datetime
import yfinance as yf
from utils.db_utils import db, GameInfo

from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3000"}})

game_bp = Blueprint('game', __name__)
RECORDS_DIR = 'records'
PREDICTIONS_DIR = 'predictions'


@game_bp.route('/game_info', methods=['GET'])
@cross_origin()
def get_game_info():
    game_id = request.args.get('game_id')
    user_id = request.args.get('user_id')
    if not game_id or not user_id:
        return jsonify({'error': 'Game ID and User ID are required'}), 400

    game_infos = GameInfo.query.filter_by(game_id=game_id, user_id=user_id).all()
    game_info_list = [
        {
            'id': game_info.id,
            'game_id': game_info.game_id,
            'user_id': game_info.user_id,
            'cash': game_info.cash,
            'portfolio_value': game_info.portfolio_value,
            'total_assets': game_info.total_assets,
            'stocks': game_info.stocks,
            'score': game_info.score,
            'last_updated': game_info.last_updated.isoformat()
        }
        for game_info in game_infos
    ]
    # 倒序排列
    game_info_list.sort(key=lambda x: x['last_updated'], reverse=True)
    return jsonify(game_info_list)


@game_bp.route('/game_info', methods=['POST'])
@cross_origin()
def store_game_info():
    game_info_data = request.get_json()
    print("game_info_data:", game_info_data)
    game_info = GameInfo(
        game_id=game_info_data['game_id'],
        user_id=game_info_data['user_id'],
        cash=game_info_data['cash'],
        portfolio_value=game_info_data['portfolio_value'],
        total_assets=game_info_data['cash'] + game_info_data['portfolio_value'],
        stocks=game_info_data['stocks'],
        score=game_info_data['score'],
        last_updated=datetime.utcnow()
    )
    db.session.add(game_info)
    db.session.commit()
    return jsonify({'status': 'success'})


@game_bp.route('/predictions/<stock>', methods=['GET'])
@cross_origin()
def get_predictions(stock):
    file_path = os.path.join(PREDICTIONS_DIR, f'{stock}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            predictions = pd.read_pickle(f)
        return predictions.to_json(orient='records')
    else:
        return jsonify({'error': 'Prediction file not found'}), 404


@game_bp.route('/next_trading_day', methods=['POST'])
@cross_origin()
def next_trading_day():
  data = request.json
  current_date = pd.to_datetime(data['current_date'])
  n = data.get('n', 1)

  # 获取所有交易日数据
  df = yf.download("SPY", start="2015-01-01", end="2025-12-31")  # 使用SPY代表标准交易日
  trading_days = df.index

  # 计算下一个交易日
  next_trading_day = trading_days[trading_days > current_date][n - 1]

  return jsonify({"next_trading_day": next_trading_day.strftime('%Y-%m-%d')})


@game_bp.route('/last_trading_day', methods=['POST'])
@cross_origin()
def last_trading_day():
  data = request.json
  current_date = pd.to_datetime(data['current_date'])
  n = data.get('n', 1)

  # 获取所有交易日数据
  df = yf.download("SPY", start="2015-01-01", end="2025-12-31")  # 使用SPY代表标准交易日
  trading_days = df.index

  # 计算上一个交易日
  previous_trading_days = trading_days[trading_days < current_date]

  # 确保存在足够的历史交易日
  if len(previous_trading_days) < n:
    return jsonify({"error": "Not enough trading days in the past"}), 400

  last_trading_day = previous_trading_days[-n]

  return jsonify({"last_trading_day": last_trading_day.strftime('%Y-%m-%d')})
