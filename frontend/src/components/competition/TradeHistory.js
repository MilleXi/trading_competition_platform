import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../../css/TradeHistory.css';  // 导入CSS文件

const TradeHistory = ({ userId, refreshHistory, selectedStock }) => {
  const [history, setHistory] = useState([]);
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/api/transactions`, {
          params: {
            user_id: userId
          }
        });
        const filteredHistory = response.data.filter(trade => selectedStock.includes(trade.stock_symbol));
        setHistory(filteredHistory);
      } catch (error) {
        console.error('Error fetching trade history:', error);
      }
    };

    fetchHistory();
  }, [userId, refreshHistory, selectedStock]);

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
          {history.length > 0 ? (
            history.map((trade, index) => (
              <tr key={index}>
                <td>{new Date(trade.date).toLocaleDateString()}</td>
                {selectedStock.map((stock, idx) => (
                  <td key={idx}>
                    {trade.stock_symbol === stock ? `${trade.transaction_type}: ${trade.amount}` : ''}
                  </td>
                ))}
                <td>{/* Calculate total value here based on trade data */}</td>
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
