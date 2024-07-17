import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../../css/TradeHistory.css';  // 导入CSS文件

const TradeHistory = ({ userId, refreshHistory, selectedStock }) => {
  const [history, setHistory] = useState({});

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/transactions', {
          params: {
            user_id: userId,
            stock_symbols: selectedStock.join(','),
          },
        });
        console.log('history response data:', response.data);
        setHistory(response.data.transactions_by_date);
      } catch (error) {
        console.error('Error fetching trade history:', error);
      }
    };

    fetchHistory();
  }, [userId, refreshHistory, selectedStock]);

  console.log('history:', history);
  console.log('selectedStock:', selectedStock);

  return (
    <div className="history-section">
      <h4>Trade History</h4>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            {selectedStock.map((stock, index) => (
              <th key={index}>{stock}</th>
            ))}
            <th>Total Value</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(history).length > 0 ? (
            Object.keys(history).map((date, index) => (
              <tr key={index}>
                <td>{new Date(date).toLocaleDateString()}</td>
                {selectedStock.map((stock, idx) => {
                  const tradeDetails = history[date][stock] || [];
                  const tradeInfo = tradeDetails.map(trade => `${trade.transaction_type}: ${trade.amount}`).join(', ');

                  return <td key={idx}>{tradeInfo}</td>;
                })}
                <td>
                  {selectedStock.reduce((total, stock) => {
                    const tradeDetails = history[date][stock] || [];
                    return total + tradeDetails.reduce((subtotal, trade) => subtotal + (trade.amount || 0), 0);
                  }, 0)}
                </td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={selectedStock.length + 2} style={{ textAlign: 'center' }}>No trade history</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default TradeHistory;
