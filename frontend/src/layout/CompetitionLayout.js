import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';
import Modal from 'react-modal';
import '../css/home.css';
import '../css/competition.css';
import { AppHeader, AppFooter, AppHeaderDropdown } from '../components/index';
import { CDropdown, CDropdownItem, CDropdownMenu, CDropdownToggle } from '@coreui/react';
import CandlestickChart from '../components/competition/CandlestickChart';
import StockTradeComponent from '../components/competition/StockTrade';
import FinancialReport from '../components/competition/FinancialReport';
import TradeHistory from '../components/competition/TradeHistory';
import axios from 'axios';

const CompetitionLayout = () => {
  const initialBalance = 100000;
  const [marketData, setMarketData] = useState([]);
  const [currentRound, setCurrentRound] = useState(1);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [buyAmount, setBuyAmount] = useState('');
  const [sellAmount, setSellAmount] = useState('');
  const [hold, setHold] = useState(false);
  const [selectedTrades, setSelectedTrades] = useState({});
  const TMinus = 60;
  const MaxRound = 10;
  const [counter, setCounter] = useState(TMinus);
  const [gameEnd, setGameEnd] = useState(false);
  const [refreshHistory, setRefreshHistory] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedTickers, setSelectedTickers] = useState([]);
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

  const tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'C', 'WFC', 'GS',
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'XOM', 'CVX', 'COP', 'SLB', 'BKR',
    'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX', 'CAT', 'DE', 'MMM', 'GE', 'HON'
  ];

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const handleTickerSelection = (ticker) => {
    setSelectedTickers((prev) => {
      if (prev.includes(ticker)) {
        return prev.filter((item) => item !== ticker);
      }
      if (prev.length < 3) {
        return [...prev, ticker];
      }
      return prev;
    });
  };

  const confirmSelection = () => {
    if (selectedTickers.length === 3) {
      setSelectedStock(selectedTickers);
      setCurrentStock(selectedTickers[0]); // 默认选择第一个股票
      closeModal();
    }
  };

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

  const handleSubmit = async () => {
  const date = currentDate.toISOString();

  for (const stock of Object.keys(selectedTrades)) {
    const { type, amount } = selectedTrades[stock];

    const transaction = {
      user_id: userId,
      stock_symbol: stock,
      transaction_type: type,
      amount: parseFloat(amount),
      date: date
    };

    try {
      await axios.post('http://localhost:5000/api/transactions', transaction);
      console.log('Transaction submitted:', transaction);
    } catch (error) {
      console.error('Error submitting transaction:', error);
    }
  }

  setBuyAmount('');
  setSellAmount('');
  setHold(false);
  setSelectedTrades({});
  setRefreshHistory(!refreshHistory); // 触发交易历史刷新
  handleNextRound();
};


  const handleNextRound = () => {
    // 更新游戏日期逻辑，每次增加n个交易日
    const nextDate = new Date(currentDate);
    const n = 1; // 设定n为1个交易日
    nextDate.setDate(currentDate.getDate() + n);

    if (currentRound === MaxRound) {
      setGameEnd(true);
      return;
    }

    setCurrentRound(currentRound + 1);
    setCurrentDate(nextDate);
    setCounter(TMinus);
  };

  const filteredMarketData = marketData.filter(data => data.date <= currentDate);

  return (
    <div className="background">
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100" style={{ color: 'white' }}>
          <AppHeader />
          <div className="top-bar d-flex justify-content-between align-items-center">
            <div>Mode: {difficulty}</div>
            <button onClick={openModal}>Select Stocks</button>
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
              {selectedStock.map((stock) => (
                <button key={stock} onClick={() => setCurrentStock(stock)}>{stock}</button>
              ))}
            </div>
            <div className="market-display d-flex" style={{ flexDirection: 'row', alignItems: 'end' }}>
              <div className="stock-info" style={{ backgroundColor: 'transparent', flex: '1', padding: '1em' }}>
                <div style={{ backgroundColor: 'white', color: 'black' }}>
                  <CandlestickChart data={CandlestickChartData} stockName={selectedStock}/>
                </div>
              </div>
              <div className="report" style={{ flex: "1", padding: '1em' }}>
                <FinancialReport selectedStock={selectedStock}
                  currentDate={currentDate}
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
                  rowsPerPage={5} />
              </div>
            </div>
            <div className="bottom-section d-flex justify-content-between">
              <StockTradeComponent initialBalance={initialBalance} userId={userId} selectedStock={selectedStock} />
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
              <TradeHistory userId={userId} refreshHistory={refreshHistory} selectedStock={selectedStock} />            </div>
          </div>
        </div>
      </div>

      <Modal isOpen={isModalOpen} onRequestClose={closeModal} contentLabel="Select Stocks">
        <h2>Select 3 Stocks</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap' }}>
          {tickers.map((ticker) => (
            <button
              key={ticker}
              onClick={() => handleTickerSelection(ticker)}
              style={{
                margin: '5px',
                padding: '10px',
                backgroundColor: selectedTickers.includes(ticker) ? 'green' : 'gray'
              }}
            >
              {ticker}
            </button>
          ))}
        </div>
        <button onClick={confirmSelection}>Confirm</button>
        <button onClick={closeModal}>Close</button>
      </Modal>
    </div>
  );
}

export default CompetitionLayout;
