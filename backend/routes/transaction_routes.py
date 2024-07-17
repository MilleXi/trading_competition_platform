import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, Blueprint, jsonify, request, current_app
from flask_cors import CORS, cross_origin
from utils.db_utils import db, Transaction
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "http://localhost:3000"}})

transaction_bp = Blueprint('transaction', __name__)


# 获取所有交易记录
@transaction_bp.route('/transactions', methods=['GET'])
@cross_origin()
def get_transactions():
    print('arg', request.args)
    user_id = request.args.get('user_id')
    game_id = request.args.get('game_id')
    stock_symbols = request.args.get('stock_symbols')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    query = Transaction.query

    if user_id:
        query = query.filter_by(user_id=user_id)

    if game_id:
        query = query.filter_by(game_id=game_id)

    if stock_symbols:
        stock_symbol_list = stock_symbols.split(',')
        query = query.filter(Transaction.stock_symbol.in_(stock_symbol_list))

    if start_date:
        query = query.filter(Transaction.date >= start_date)

    if end_date:
        query = query.filter(Transaction.date <= end_date)

    transactions = query.all()

    # Organize data
    transaction_data = {
        'user_id': user_id,
        'game_id': game_id,
        'transactions_by_date': defaultdict(lambda: defaultdict(list))
    }

    for transaction in transactions:
        date_str = transaction.date.strftime('%Y-%m-%d')
        transaction_data['transactions_by_date'][date_str][transaction.stock_symbol].append({
            'id': transaction.id,
            'transaction_type': transaction.transaction_type,
            'amount': transaction.amount,
        })

    # Sort by date, most recent first
    sorted_transaction_data = {
        'user_id': user_id,
        'game_id': game_id,
        'transactions_by_date': dict(sorted(transaction_data['transactions_by_date'].items(),
                                            key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'),
                                            reverse=True))
    }

    print(sorted_transaction_data)

    return jsonify(sorted_transaction_data)


# 创建新的交易记录
@transaction_bp.route('/transactions', methods=['POST'])
@cross_origin()
def create_transaction():
    data = request.get_json()
    try:
        date = datetime.strptime(data['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        return jsonify({"error": "Incorrect date format"}), 400

    new_transaction = Transaction(
        game_id=data['game_id'],
        user_id=data['user_id'],
        stock_symbol=data['stock_symbol'],
        transaction_type=data['transaction_type'],
        amount=data['amount'],
        date=date
    )
    db.session.add(new_transaction)
    db.session.commit()
    return jsonify({'message': 'Transaction created successfully'}), 201


# 更新交易记录
@transaction_bp.route('/transactions/<int:transaction_id>', methods=['PUT'])
@cross_origin()
def update_transaction(transaction_id):
    data = request.get_json()
    transaction = Transaction.query.get_or_404(transaction_id)
    try:
        date = datetime.strptime(data['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        return jsonify({"error": "Incorrect date format"}), 400

    transaction.game_id = data['game_id']
    transaction.user_id = data['user_id']
    transaction.stock_symbol = data['stock_symbol']
    transaction.transaction_type = data['transaction_type']
    transaction.amount = data['amount']
    transaction.date = date
    db.session.commit()
    return jsonify({'message': 'Transaction updated successfully'})


# 删除交易记录
@transaction_bp.route('/transactions/<int:transaction_id>', methods=['DELETE'])
@cross_origin()
def delete_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    db.session.delete(transaction)
    db.session.commit()
    return jsonify({'message': 'Transaction deleted successfully'})

app.register_blueprint(transaction_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(port=8000)
