import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FinancialReport = ({ selectedStock }) => {
  const [stockData, setStockData] = useState([]);

  useEffect(() => {
    console.log('Fetching data for:', selectedStock)
    const fetchStockData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/stored_stock_data', {
          params: {
            symbol: selectedStock,
            start_date: '2023-01-01',
            end_date: '2023-01-09'
          }
        });
        console.log("Received data:", response.data);  // 打印调试信息
        setStockData(response.data);
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
  }, [selectedStock]);

  return (
    <div className="financial-report">
      <h3>Financial Report for {selectedStock}</h3>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Open</th>
            <th>High</th>
            <th>Low</th>
            <th>Close</th>
            <th>Volume</th>
            <th>MA5</th>
            <th>MA10</th>
            <th>MA20</th>
            <th>RSI</th>
            <th>MACD</th>
            <th>VWAP</th>
            <th>SMA</th>
            <th>Std Dev</th>
            <th>Upper Band</th>
            <th>Lower Band</th>
            <th>ATR</th>
            <th>Sharpe Ratio</th>
            <th>Beta</th>
          </tr>
        </thead>
        <tbody>
          {stockData.map((data) => (
            <tr key={data.date}>
              <td>{new Date(data.date).toLocaleDateString()}</td>
              <td>{data.open}</td>
              <td>{data.high}</td>
              <td>{data.low}</td>
              <td>{data.close}</td>
              <td>{data.volume}</td>
              <td>{data.ma5}</td>
              <td>{data.ma10}</td>
              <td>{data.ma20}</td>
              <td>{data.rsi}</td>
              <td>{data.macd}</td>
              <td>{data.vwap}</td>
              <td>{data.sma}</td>
              <td>{data.std_dev}</td>
              <td>{data.upper_band}</td>
              <td>{data.lower_band}</td>
              <td>{data.atr}</td>
              <td>{data.sharpe_ratio}</td>
              <td>{data.beta}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default FinancialReport;
