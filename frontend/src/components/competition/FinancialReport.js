import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FinancialReport = ({ selectedStock }) => {
  const [stockData, setStockData] = useState([]);

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/stored_stock_data', {
          params: {
            symbol: selectedStock,
            start_date: '2023-01-01',
            end_date: '2023-12-31'
          }
        });
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
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default FinancialReport;
