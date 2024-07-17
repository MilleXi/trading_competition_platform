import React, { useState, useEffect, useRef } from 'react';
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
import { v4 as uuidv4 } from 'uuid';
import App from '../App';
import zIndex from '@mui/material/styles/zIndex';
import Box from '@mui/material/Box';
import Checkbox from '@mui/material/Checkbox';
import FormControl from '@mui/material/FormControl';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormLabel from '@mui/material/FormLabel';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';

const CompetitionLayout = () => {
  const initialBalance = 100000;
  const startDate = new Date('2023-01-03');
  const gameIdRef = useRef(uuidv4());
  const gameId = gameIdRef.current;
  const modelList = ['LSTM']
  const [currentRound, setCurrentRound] = useState(1);
  const [currentDate, setCurrentDate] = useState(startDate);
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [selectedStockList, setSelectedStockList] = useState(['AAPL', 'MSFT', 'GOOGL']);
  const [stockData, setStockData] = useState([]);
  const [selectedTrades, setSelectedTrades] = useState(
    selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: { type: 'hold', amount: '0' } }), {})
  );
  const TMinus = 60;
  const MaxRound = 10;
  const [counter, setCounter] = useState(TMinus);
  const [gameEnd, setGameEnd] = useState(false);
  const [refreshHistory, setRefreshHistory] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(true);
  const [selectedTickers, setSelectedTickers] = useState([]);
  const userId = 1;
  const [CandlestickChartData, setCandlestickChartData] = useState([]);
  const location = useLocation();
  const { difficulty } = location.state || { difficulty: 'Easy' };
  const rootElement = document.getElementById('root');
  const [aiStrategy, setAiStrategy] = useState({});
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [stopCounter, setStopCounter] = useState(false);

  Modal.setAppElement(rootElement);

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
          params: {
            symbol: selectedStock,
            start_date: '2021-01-01',
            end_date: '2024-01-01'
          }
        });
        setStockData(response.data);
        setCandlestickChartData(response.data.map(data => ({
          date: new Date(data.date),
          open: parseFloat(data.open),
          high: parseFloat(data.high),
          low: parseFloat(data.low),
          close: parseFloat(data.close),
        })));
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };

    fetchStockData();
  }, [selectedStock]);

  console.log('stockData:', stockData);
  console.log('selectedStock:', selectedStock);
  console.log('CandlestickChartData:', CandlestickChartData);

  useEffect(() => {
    const timerId = setTimeout(() => {
      if (counter > 0 && !stopCounter) {
        setCounter(counter - 1);
      } else if (counter === 0 && !gameEnd) {
        if (Object.keys(selectedTrades).length > 0)
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

  const confirmSelection = async () => {
    if (selectedTickers.length === 3) {
      setSelectedStockList(selectedTickers);
      setSelectedStock(selectedTickers[0]); // 默认选择第一个股票
      closeModal();
    } else
      return;

    try {
      const response = await axios.post('http://localhost:8000/api/run_strategy', {
        tickers: selectedTickers,
        game_id: gameId,
      });
      console.log('Strategy result:', response.data);

      // 保存结果到数据库
      for (const record of response.data.trade_log) {
        const logEntry = {
          ...record,
          model: "LSTM",
          game_id: gameId
        };
        await axios.post('http://localhost:8000/api/save_trade_log', logEntry);
      }

    } catch (error) {
      console.error('Error running strategy:', error);
    }
  };

  const handleSubmit = async () => {
    console.log('handleSubmit:');

    const date = currentDate.toISOString();

    for (const stock of Object.keys(selectedTrades)) {
      const { type, amount } = selectedTrades[stock];

      const transaction = {
        game_id: gameId,
        user_id: userId,
        stock_symbol: stock,
        transaction_type: type,
        amount: parseFloat(amount),
        date: date,
      };

      try {
        await axios.post('http://localhost:8000/api/transactions', transaction);
        console.log('Transaction submitted:', transaction);
      } catch (error) {
        console.error('Error submitting transaction:', error);
      }
    }

    // 获取AI策略
    try {
      const aiResponse = await axios.get('http://localhost:8000/api/get_trade_log', {
        params: {
          game_id: gameId,
          model: 'LSTM',
          date: date.split('T')[0],
        }
      });
      // 传输ai的交易记录
      if (aiResponse.data) {
        console.log("AI Strategy:", aiResponse.data)
        setAiStrategy(aiResponse.data);

        const aiTransactions = Object.entries(aiResponse.data.change).map(([stock, amount]) => ({
          game_id: gameId,
          user_id: 'ai', // 或者用一个特定的AI用户ID
          stock_symbol: stock,
          transaction_type: amount > 0 ? 'buy' : 'sell',
          amount: Math.abs(amount),
          date: date,
        }));
        for (const transaction of aiTransactions) {
          try {
            await axios.post('http://localhost:8000/api/transactions', transaction);
            console.log('AI Transaction submitted:', transaction);
          } catch (error) {
            console.error('Error submitting AI transaction:', error);
          }
        }

        setShowStrategyModal(true); // 显示策略模态窗口
      } else {
        console.error('No AI strategy found');
      }
    } catch (error) {
      console.error('Error fetching AI strategy:', error);
    }

    setStopCounter(true);

  };


  const handleNextRound = async () => {
    // 更新游戏日期逻辑，每次增加n个交易日
    const n = 1; // 设定n为1个交易日
    try {
      const response = await axios.post('http://localhost:8000/api/next_trading_day', {
        current_date: currentDate.toISOString().split('T')[0],
        n: n
      });
      const nextDate = new Date(response.data.next_trading_day);

      if (currentRound === MaxRound) {
        setGameEnd(true);
        setStopCounter(true);
        return;
      }

      setCurrentRound(currentRound + 1);
      setCurrentDate(nextDate);
      setCounter(TMinus);
    } catch (error) {
      console.error('Error fetching next trading day:', error);
    }
  };

  const filteredCandlestickChartData = CandlestickChartData.filter(data => data.date < currentDate);

  const closeStrategyModal = () => {
    setShowStrategyModal(false);
    setSelectedTrades(selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: { type: 'hold', amount: '0' } }), {}));
    setRefreshHistory(prev => !prev); // 触发交易历史刷新
    handleNextRound();
    setStopCounter(false);
  };

  return (
    <div className="background">
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100" style={{ color: 'white' }}>
          <AppHeader />
          <div className="d-flex justify-content-between align-items-center">
          <div
            onClick={openModal}
            style={{
                backgroundColor: 'green',
                color: 'white',
                padding: '10px 20px',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                textAlign: 'center',
                fontSize: '16px',
                display: 'inline-block'
            }}
            >
            Select Stocks
          </div>
          <div>Mode: {difficulty}  &emsp;  Current Round: {currentRound}/{MaxRound}  &emsp;  Current Date: {currentDate.toISOString().split('T')[0]}  &emsp;  Countdown: {counter}</div>
            <CDropdown variant="dropdown">
              <CDropdownToggle caret={true}>
                <span style={{ color: 'white' }}>Game Credits: 100</span>
              </CDropdownToggle>

              <CDropdownMenu className='dropdown-menu'>
                <CDropdownItem className='dropdown-item'>
                  <span style={{ color: 'white' }}>Shop</span>
                </CDropdownItem>
                {/* 积分商城内容可以在此处添加 */}
              </CDropdownMenu>
            </CDropdown>
          </div>

          <div className="body flex-grow-1 px-3 d-flex flex-column align-items-center">
            <div className="d-flex justify-content-center w-100 mb-3" style={{ padding: '1em' }}>
              {selectedStockList.map((stock) => (
                <button key={stock} onClick={() => setSelectedStock(stock)}>{stock}</button>
              ))}
            </div>
            <div className="market-display d-flex" style={{ flexDirection: 'row', alignItems: 'end' }}>
              <div className="stock-info" style={{ backgroundColor: 'transparent', flex: '1', padding: '1em' }}>
                <div style={{ backgroundColor: 'white', color: 'black' }}>
                  <CandlestickChart data={filteredCandlestickChartData} stockName={selectedStock} style={{ zIndex: '1' }} />
                </div>
              </div>
              <div className="report" style={{ flex: "1", padding: '1em' }}>
                <FinancialReport selectedStock={selectedStock}
                  currentDate={currentDate}
                  stockData={stockData}
                  setStockData={setStockData}
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
              <StockTradeComponent selectedTrades={selectedTrades} setSelectedTrades={setSelectedTrades}
                initialBalance={initialBalance}
                userId={userId}
                selectedStock={selectedStockList}
                handleSubmit={handleSubmit} />
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
              <TradeHistory userId={userId} refreshHistory={refreshHistory} selectedStock={selectedStockList} gameId={gameId} />
            </div>
          </div>
        </div>
      </div>
    
      <Modal isOpen={isModalOpen} onRequestClose={closeModal} contentLabel="Select Stocks"
        style={{
            content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '80%',
            height: 'auto',
            zIndex: '1000',
            border: '1px solid #ccc',
            borderRadius: '10px',
            padding: '20px',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
            }
        }}
        >
        <h2 style={{ textAlign: 'center', color: '#333', marginBottom: '20px'}}>Please Select 3 Stocks</h2>
        <Grid container spacing={2}>
            {tickers.map((ticker) => (
            <Grid item xs={2} key={ticker}>
                <FormControlLabel
                control={
                    <Checkbox
                    checked={selectedTickers.includes(ticker)}
                    onChange={() => handleTickerSelection(ticker)}
                    name={ticker}
                    color="primary"
                    />
                }
                label={
                    <Typography sx={{ fontSize: '22px', color: 'black' }}>
                      {ticker}
                    </Typography>
                  }
                />
            </Grid>
            ))}
        </Grid>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px' }}>
            <button 
            onClick={confirmSelection}
            style={{
                padding: '10px 20px',
                backgroundColor: '#008CBA',
                color: '#fff',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                outline: 'none',
                fontSize: '20px'
            }}
            >
            Confirm
            </button>
        </div>
      </Modal>


      <Modal
        isOpen={showStrategyModal}
        onRequestClose={closeStrategyModal}
        contentLabel="Strategy Modal"
        style={{
          content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '80%',
            height: 'auto',
            zIndex: '1000'
          }
        }}
      >
        <h2>Strategy for {currentDate.toISOString().split('T')[0]}</h2>
        <div>
          <h3>Player's Strategy:</h3>
          {Object.entries(selectedTrades).map(([stock, trade]) => (
            <div key={stock}>
              <p>{stock}: {trade.type} {trade.amount}</p>
            </div>
          ))}
        </div>
        <div>
          <h3>AI's Strategy:</h3>
          {aiStrategy && aiStrategy.change ? (
            Object.entries(aiStrategy.change).map(([stock, change]) => (
              <div key={stock}>
                <p>{stock}: {change}</p>
              </div>
            ))
          ) : (
            <p>No AI strategy found</p>
          )}
        </div>
        <button onClick={closeStrategyModal} style={{ display: 'block', margin: '20px auto' }}>Close</button>
      </Modal>
    </div >
  );
}

export default CompetitionLayout;
