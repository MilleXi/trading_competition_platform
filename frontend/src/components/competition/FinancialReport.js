import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';


const FinancialReport = ({
  selectedStock,
  currentDate,
  chartWidth = '100%',
  chartHeight = 300,
  chartTop = 50,
  chartLeft = 0,
  chartRight = 0,
  chartBackgroundOpacity = 0.9,
  chartTitleColor = 'black',
  backgroundColor = 'rgba(255, 255, 255, 0.9)',
  chartPaddingLeft = 50,
  rowGap = 10, // default row gap is 10px
  colGap = 10, // default col gap is 10px
  chartContainerHeight = 600, // height of the chart container
  rowsPerPage = 5 // number of rows to display per page
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
            start_date: '2021-01-01',
            end_date: currentDate.toISOString().split('T')[0]
          }
        });
        setStockData(response.data.reverse());
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
  }, [selectedStock]);

  const handleAttributeClick = (attribute) => {
    if (attribute === 'Date') return;
    if (selectedAttribute === attribute) {
      setShowChart(false);
      setSelectedAttribute(null);
      return;
    }
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
      date: new Date(data.date).toLocaleDateString('en-CA'),
      [selectedAttribute]: data[selectedAttribute?.replace(' ', '_').toLowerCase()]
    }));
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-CA');
  };

  const attributeLabels = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
    'VWAP', 'SMA', 'Std Dev', 'Upper Band', 'Lower Band', 'ATR', 'Sharpe Ratio', 'Beta'
  ];

  const firstRowAttributes = attributeLabels.slice(0, Math.ceil(attributeLabels.length / 2));
  const secondRowAttributes = ['Date', ...attributeLabels.slice(Math.ceil(attributeLabels.length / 2))];

  return (
    <div className="financial-report" style={{ position: 'relative' }}>
      <h3 style={{ textAlign: 'center' }}>Financial Report for {selectedStock}</h3>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ paddingBottom: '0.5em' }}>
          <div style={{ overflowY: 'auto', maxHeight: rowsPerPage * 30 + 'px' }}>
            <table style={{ borderCollapse: 'separate', borderSpacing: `${rowGap}px ${colGap}px` }}>
              <thead>
                <tr>
                  {firstRowAttributes.map(attribute => (
                    <th key={attribute} className="attribute-name" onClick={() => handleAttributeClick(attribute)}>{attribute}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stockData.slice(0, rowsPerPage).map(data => (
                  <tr key={`${data.date}-first-row`}>
                    {firstRowAttributes.map(attribute => (
                      <td key={attribute}>{attribute === 'Date' ? formatDate(data.date) : data[attribute.toLowerCase().replace(' ', '_')]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div style={{ paddingTop: '0.5em', maxHeight: '300px' }}>
          <div style={{ overflowY: 'auto', maxHeight: '300px' }}>
            <table style={{ borderCollapse: 'separate', borderSpacing: `${rowGap}px ${colGap}px` }}>
              <thead>
                <tr>
                  {secondRowAttributes.map(attribute => (
                    <th key={attribute} className="attribute-name" onClick={() => handleAttributeClick(attribute)}>{attribute}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stockData.slice(0, rowsPerPage).map(data => (
                  <tr key={`${data.date}-second-row`}>
                    {secondRowAttributes.map(attribute => (
                      <td key={attribute}>{attribute === 'Date' ? formatDate(data.date) : data[attribute.toLowerCase().replace(' ', '_')]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      {
        showChart && selectedAttribute && (
          <>
            <div className="chart-container-wrapper" style={{
              position: 'absolute',
              top: `${chartTop}px`,
              left: `${chartLeft}px`,
              right: `${chartRight}px`,
              zIndex: 1,
              backgroundColor: backgroundColor,
              // padding: '10px',
              width: '100%',
              paddingLeft: `${chartPaddingLeft}px`,
              color: 'black'
            }}>
              <h4 style={{ textAlign: 'center', color: chartTitleColor }}>Historical trends of {selectedAttribute}</h4>
              <div className="chart-container" style={{ width: chartWidth, height: `${chartContainerHeight}px` }}>
                <ResponsiveContainer>
                  <LineChart data={getChartData()} margin={{ top: 5, right: 30, left: 50, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey={selectedAttribute} stroke="#8884d8" strokeWidth={2} activeDot={{ r: 0 }} dot={{ r: 0 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )
      }
    </div>
  );
};

export default FinancialReport;
