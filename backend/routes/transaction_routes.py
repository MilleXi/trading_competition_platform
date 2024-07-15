import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Blueprint, jsonify, request, current_app
from utils.db_utils import db, Transaction
from datetime import datetime

transaction_bp = Blueprint('transaction', __name__)


# 获取所有交易记录
@transaction_bp.route('/transactions', methods=['GET'])
def get_transactions():
    user_id = request.args.get('user_id')
    stock_symbol = request.args.get('stock_symbol')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    query = Transaction.query
    if user_id:
        query = query.filter_by(user_id=user_id)
    if stock_symbol:
        query = query.filter_by(stock_symbol=stock_symbol)
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)

    transactions = query.all()
    transaction_data = [{
        'id': transaction.id,
        'user_id': transaction.user_id,
        'stock_symbol': transaction.stock_symbol,
        'transaction_type': transaction.transaction_type,
        'amount': transaction.amount,
        'date': transaction.date
    } for transaction in transactions]
    return jsonify(transaction_data)


# 创建新的交易记录
@transaction_bp.route('/transactions', methods=['POST'])
def create_transaction():
    data = request.get_json()
    new_transaction = Transaction(
        user_id=data['user_id'],
        stock_symbol=data['stock_symbol'],
        transaction_type=data['transaction_type'],
        amount=data['amount'],
        date=datetime.strptime(data['date'], '%Y-%m-%d %H:%M:%S')
    )
    db.session.add(new_transaction)
    db.session.commit()
    return jsonify({'message': 'Transaction created successfully'}), 201


# 更新交易记录
@transaction_bp.route('/transactions/<int:transaction_id>', methods=['PUT'])
def update_transaction(transaction_id):
    data = request.get_json()
    transaction = Transaction.query.get_or_404(transaction_id)
    transaction.user_id = data['user_id']
    transaction.stock_symbol = data['stock_symbol']
    transaction.transaction_type = data['transaction_type']
    transaction.amount = data['amount']
    transaction.date = datetime.strptime(data['date'], '%Y-%m-%d %H:%M:%S')
    db.session.commit()
    return jsonify({'message': 'Transaction updated successfully'})


# 删除交易记录
@transaction_bp.route('/transactions/<int:transaction_id>', methods=['DELETE'])
def delete_transaction(transaction_id):
    transaction = Transaction.query.get_or_404(transaction_id)
    db.session.delete(transaction)
    db.session.commit()
    return jsonify({'message': 'Transaction deleted successfully'})
