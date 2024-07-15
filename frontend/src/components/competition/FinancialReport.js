import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const FinancialReport = ({
  selectedStock,
  chartWidth = '100%',
  chartHeight = 300,
  chartTop = 50,
  chartLeft = 0,
  chartRight = 0,
  chartBackgroundOpacity = 0.9,
  chartTitleColor = 'black',
  backgroundColor = 'rgba(255, 255, 255, 0.9)',
  chartPaddingLeft = 50
}) => {
  const [stockData, setStockData] = useState([]);
  const [selectedAttribute, setSelectedAttribute] = useState(null);
  const [showChart, setShowChart] = useState(false);

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/stored_stock_data', {
          params: {
            symbol: selectedStock,
            start_date: '2023-01-01',
            end_date: '2023-01-09'
          }
        });
        setStockData(response.data);
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
  }, [selectedStock]);

  const handleAttributeClick = (attribute) => {
    setSelectedAttribute(attribute);
    setShowChart(true);
  };

  const handleOutsideClick = (e) => {
    if (e.target.closest('.attribute-name') === null && e.target.closest('.chart-container') === null) {
      setShowChart(false);
    }
  };

  useEffect(() => {
    if (showChart) {
      document.addEventListener('click', handleOutsideClick);
    } else {
      document.removeEventListener('click', handleOutsideClick);
    }
    return () => {
      document.removeEventListener('click', handleOutsideClick);
    };
  }, [showChart]);

  const getChartData = () => {
    return stockData.map(data => ({
      date: new Date(data.date).toLocaleDateString(),
      value: data[selectedAttribute]
    }));
  };

  const attributeLabels = {
    open: 'Open Price',
    high: 'High Price',
    low: 'Low Price',
    close: 'Close Price',
    volume: 'Volume',
    ma5: 'MA5',
    ma10: 'MA10',
    ma20: 'MA20',
    rsi: 'RSI',
    macd: 'MACD',
    vwap: 'VWAP',
    sma: 'SMA',
    std_dev: 'Standard Deviation',
    upper_band: 'Upper Band',
    lower_band: 'Lower Band',
    atr: 'ATR',
    sharpe_ratio: 'Sharpe Ratio',
    beta: 'Beta'
  };

  return (
    <div className="financial-report" style={{ position: 'relative' }}>
      <h3 style={{ textAlign: 'center' }}>Financial Report for {selectedStock}</h3>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('open')}>Open</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('high')}>High</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('low')}>Low</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('close')}>Close</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('volume')}>Volume</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('ma5')}>MA5</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('ma10')}>MA10</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('ma20')}>MA20</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('rsi')}>RSI</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('macd')}>MACD</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('vwap')}>VWAP</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('sma')}>SMA</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('std_dev')}>Std Dev</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('upper_band')}>Upper Band</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('lower_band')}>Lower Band</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('atr')}>ATR</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('sharpe_ratio')}>Sharpe Ratio</th>
            <th className="attribute-name" onClick={() => handleAttributeClick('beta')}>Beta</th>
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
      {showChart && selectedAttribute && (
        <>
          <div style={{ position: 'absolute', top: `${chartTop}px`, left: `${chartLeft}px`, right: `${chartRight}px`, zIndex: 1, backgroundColor: backgroundColor, padding: '10px', paddingLeft: `${chartPaddingLeft}px` }}>
            <h4 style={{ textAlign: 'center', color: chartTitleColor }}>Historical trends of {attributeLabels[selectedAttribute]}</h4>
            <div className="chart-container" style={{ width: chartWidth, height: `${chartHeight}px` }}>
              <ResponsiveContainer>
                <LineChart data={getChartData()} margin={{ top: 5, right: 30, left: 50, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#8884d8" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default FinancialReport;
