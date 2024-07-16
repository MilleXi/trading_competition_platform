import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import '../css/home.css';
import '../css/competition.css';
import { AppHeader, AppFooter, AppHeaderDropdown } from '../components/index';
import { CDropdown, CDropdownItem, CDropdownMenu, CDropdownToggle } from '@coreui/react';
import CandlestickChart from '../components/competition/CandlestickChart';
import StockTradeComponent from '../components/competition/StockTrade';
import FinancialReport from '../components/competition/FinancialReport';
import axios from 'axios';

const CompetitionLayout = () => {
  const initialBalance = 100000;
  const [stockData, setStockData] = useState([]);
  const [currentRound, setCurrentRound] = useState(10);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [buyAmount, setBuyAmount] = useState('');
  const [sellAmount, setSellAmount] = useState('');
  const [hold, setHold] = useState(false);
  const TMinus = 2;
  const MaxRound = 10;
  const [counter, setCounter] = useState(TMinus);
  const [gameEnd, setGameEnd] = useState(false);
  const userId = 1;
  const [CandlestickChartData, setCandlestickChartData] = useState([]);
  const location = useLocation();
  const { difficulty } = location.state || { difficulty: 'Easy' };

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/stored_stock_data', {
          params: {
            symbol: selectedStock,
            start_date: '2021-01-01',
            end_date: '2024-01-01'
          }
        });
        setStockData(response.data);
        setCandlestickChartData(response.data.map(data => ({
          x: new Date(data.date),
          y: [data.open, data.high, data.low, data.close]
        })));
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
  }, [selectedStock]);

  // console.log('stockData:', stockData);
  // console.log('selectedStock:', selectedStock);
  // console.log('CandlestickChartData:', CandlestickChartData);

  useEffect(() => {
    const timerId = setTimeout(() => {
      if (counter > 0 && !gameEnd) {
        setCounter(counter - 1);
      } else if (counter === 0 && !gameEnd) {
        if (buyAmount || sellAmount || hold)
          handleSubmit();
        else
          handleNextRound();
      }
    }, 1000);
    return () => clearTimeout(timerId);
  }, [counter, gameEnd]);

  const stockList = ['AAPL', 'GOOGL', 'AMZN'];

  const handleBuySellChange = (stock, type, amount) => {
    setSelectedTrades((prevTrades) => ({
      ...prevTrades,
      [stock]: { type, amount }
    }));
  };

  const handleHoldChange = (stock) => {
    setSelectedTrades((prevTrades) => ({
      ...prevTrades,
      [stock]: { type: 'hold', amount: '' }
    }));
  };

  const handleClear = () => {
    setSelectedTrades({});
    setBuyAmount('');
    setSellAmount('');
    setHold(false);
  };

  const handleSubmit = () => {
    // 提交逻辑，例如更新余额和股票状态
    if (buyAmount || sellAmount || hold) {
      // Handle submit logic here

      console.log('Submitted');
      // Reset form
      setBuyAmount('');
      setSellAmount('');
      setHold(false);
      handleNextRound();
    }
  };

  // console.log('counter', counter, 'currentRound', currentRound);

  const handleNextRound = () => {
    // Handle next round logic here
    // console.log('Next round');
    if (currentRound === 10) {
      setGameEnd(true);
      return;
    }
    setCurrentRound(currentRound + 1);
    setCounter(TMinus);
  };

  return (
    <div className="background">
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100" style={{ color: 'white' }}>
          <AppHeader />
          <div className="top-bar d-flex justify-content-between align-items-center">
            <div>Mode: {difficulty}</div>
            <div>Current Round: {currentRound}/{MaxRound}  &emsp;  Countdown: {counter}</div>

            <CDropdown variant="dropdown">
              <CDropdownToggle caret={true}>
                <span style={{ color: 'white' }} >Game Credits: 100</span>
              </CDropdownToggle>

              <CDropdownMenu className='dropdown-menu'>
                <CDropdownItem className='dropdown-item'>
                  <span style={{ color: 'white' }} >Shop</span>
                </CDropdownItem>
                {/* 积分商城内容可以在此处添加 */}
              </CDropdownMenu>
            </CDropdown>
          </div>
          <div className="body flex-grow-1 px-3 d-flex flex-column align-items-center">
            <div className="stock-switcher">
              <button onClick={() => setSelectedStock('AAPL')}>AAPL</button>
              <button onClick={() => setSelectedStock('GOOGL')}>GOOGL</button>
              <button onClick={() => setSelectedStock('AMZN')}>AMZN</button>
            </div>
            <div className="market-display d-flex" style={{ flexDirection: 'row', alignItems: 'end' }}>
              <div className="stock-info" style={{ backgroundColor: 'transparent', flex: '1', padding: '1em' }}>
                <div style={{ backgroundColor: 'white', color: 'black' }}>
                  <CandlestickChart data={CandlestickChartData} stockName={selectedStock}/>
                </div>
              </div>
              <div className="report" style={{ flex: "1", padding: '1em' }}>
                <FinancialReport selectedStock={selectedStock}
                  stockData={stockData}
                  chartWidth="95%"
                  chartHeight={250}
                  chartTop={80}
                  chartLeft={0}
                  chartRight={10}
                  titleColor="blue"
                  backgroundColor="rgba(255, 255, 255, 1)"
                  chartPaddingLeft={-20}
                  rowGap={20}
                  colGap={5}
                  chartContainerHeight={300}
                  rowsPerPage={6} />
              </div>
            </div>
            <div className="bottom-section d-flex justify-content-between">
              <StockTradeComponent initialBalance={initialBalance} userId={userId} />
              <div className="ranking">
                <h3>Standings:</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Competitor</th>
                      <th>Income</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>1</td>
                      <td>AI</td>
                      <td>+2000</td>
                    </tr>
                    <tr>
                      <td>2</td>
                      <td>YOU</td>
                      <td>-2000</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div className="history-section">
              <select>
                <option value="1">1</option>
                {/* Other options */}
              </select>
              <div className="history-records">
                <div>
                  <h4>yours:</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Stocks</th>
                        <th>Buy</th>
                        <th>Sell</th>
                        <th>Hold</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Stock 1</td>
                        <td>0</td>
                        <td>0</td>
                        <td>0</td>
                      </tr>
                      <tr>
                        <td>Stock 2</td>
                        <td>0</td>
                        <td>0</td>
                        <td>0</td>
                      </tr>
                      <tr>
                        <td>Stock 3</td>
                        <td>0</td>
                        <td>0</td>
                        <td>0</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div>
                  <h4>AI:</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Stocks</th>
                        <th>Buy</th>
                        <th>Sell</th>
                        <th>Hold</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Stock 1</td>
                        <td>0</td>
                        <td>0</td>
                        <td>0</td>
                      </tr>
                      <tr>
                        <td>Stock 2</td>
                        <td>0</td>
                        <td>0</td>
                        <td>0</td>
                      </tr>
                      <tr>
                        <td>Stock 3</td>
                        <td>0</td>
                        <td>0</td>
                        <td>0</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CompetitionLayout;
