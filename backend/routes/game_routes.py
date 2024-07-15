from flask import Blueprint, request, jsonify
import os
import json

game_bp = Blueprint('game', __name__)
RECORDS_DIR = 'records'
PREDICTIONS_DIR = 'predictions'

@game_bp.route('/game_info', methods=['GET'])
def get_game_info():
    with open(os.path.join(RECORDS_DIR, 'game_info.json'), 'r') as f:
        game_info = json.load(f)
    return jsonify(game_info)

@game_bp.route('/game_info', methods=['POST'])
def store_game_info():
    game_info = request.get_json()
    with open(os.path.join(RECORDS_DIR, 'game_info.json'), 'w') as f:
        json.dump(game_info, f)
    return jsonify({'status': 'success'})

@game_bp.route('/predictions/<stock>', methods=['GET'])
def get_predictions(stock):
    file_path = os.path.join(PREDICTIONS_DIR, f'{stock}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            predictions = pd.read_pickle(f)
        return jsonify(predictions)
    else:
        return jsonify({'error': 'Prediction file not found'}), 404
