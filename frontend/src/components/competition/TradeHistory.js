import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../../css/TradeHistory.css';  // 导入CSS文件

const TradeHistory = ({ userId, refreshHistory, selectedStock, gameId }) => {
  const [playerHistory, setPlayerHistory] = useState({});
  const [aiHistory, setAiHistory] = useState({});

  useEffect(() => {
    const fetchHistory = async () => {
      console.log("fetching history")
      try {
        const playerResponse = await axios.get('http://localhost:8000/api/transactions', {
          params: {
            user_id: userId,
            game_id: gameId,
            stock_symbols: selectedStock.join(','),
          },
        });

        const aiResponse = await axios.get('http://localhost:8000/api/transactions', {
          params: {
            user_id: 'ai', // 特定的AI用户ID或字符串“ai”
            game_id: gameId,
            stock_symbols: selectedStock.join(','),
          },
        });

        console.log('player history response data:', playerResponse.data);
        console.log('ai history response data:', aiResponse.data);

        setPlayerHistory(playerResponse.data.transactions_by_date);
        setAiHistory(aiResponse.data.transactions_by_date);
      } catch (error) {
        console.error('Error fetching trade history:', error);
      }
    };

    fetchHistory();
  }, [userId, refreshHistory, selectedStock]);

  console.log('player history:', playerHistory);
  console.log('ai history:', aiHistory);
  console.log('selectedStock:', selectedStock);

  const renderHistory = (history) => (
    Object.keys(history).length > 0 ? (
      Object.keys(history).reverse().map((date, index) => (
        <tr key={index}>
          <td>{new Date(date).toLocaleDateString()}</td>
          {selectedStock.map((stock, idx) => {
            const tradeDetails = history[date][stock] || [];
            const tradeInfo = tradeDetails.length > 0
              ? tradeDetails.map(trade => {
                const type = trade.transaction_type.charAt(0).toUpperCase() + trade.transaction_type.slice(1);
                return `${type}: ${trade.amount}`;
              }).join(', ')
              : 'Hold: 0';

            return <td key={idx}>{tradeInfo}</td>;
          })}
        </tr>
      ))
    ) : (
      <tr>
        <td colSpan={selectedStock.length + 1} style={{ textAlign: 'center' }}>No trade history</td>
      </tr>
    )
  );

  return (
    <div>
      <h4>Trade History</h4>
      <div className="trade-history">
        <div className="history-header">
          <span className="history-title">Player</span>
          <span className="history-divider"></span>
          <span className="history-title">AI</span>
        </div>
        <div className="history-tables">
          <div className="history-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  {selectedStock.map((stock, index) => (
                    <th key={index}>{stock}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {renderHistory(playerHistory)}
              </tbody>
            </table>
          </div>
          <div className="history-divider-vertical"></div>
          <div className="history-table">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  {selectedStock.map((stock, index) => (
                    <th key={index}>{stock}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {renderHistory(aiHistory)}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradeHistory;
