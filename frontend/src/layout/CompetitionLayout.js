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
import PointsStoreModal from '../components/competition/PointsStoreModal';
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
import GameEndModal from '../components/competition/GameEndModal';
import Standings from '../components/competition/Standings';
import { Button } from '@mui/material';

const CompetitionLayout = () => {
  const initialBalance = 100000;
  const [startDate, setStartDate] = useState(null); // 初始状态为 null
  const gameIdRef = useRef(uuidv4());
  const gameId = gameIdRef.current;
  const [modelList, setModelList] = useState([]);
  const [currentRound, setCurrentRound] = useState(1);
  const [currentDate, setCurrentDate] = useState(null); // 初始状态为 null
  const [selectedStock, setSelectedStock] = useState('AAPL');
  const [selectedStockList, setSelectedStockList] = useState(['AAPL', 'MSFT', 'GOOGL']);
  const [stockData, setStockData] = useState([]);
  const [selectedTrades, setSelectedTrades] = useState(
    selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: { type: 'hold', amount: '0' } }), {})
  );
  const [cash, setCash] = useState(initialBalance);
  const [portfolioValue, setPortfolioValue] = useState(0);
  const [totalAssets, setTotalAssets] = useState(initialBalance);
  const [aiCash, setAiCash] = useState(initialBalance);
  const [aiPortfolioValue, setAiPortfolioValue] = useState(0);
  const [aiTotalAssets, setAiTotalAssets] = useState(initialBalance);
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
  const { difficulty } = location.state || { difficulty: 'LSTM' };
  const rootElement = document.getElementById('root');
  const [aiStrategy, setAiStrategy] = useState({});
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [stopCounter, setStopCounter] = useState(true);
  const [showPointsStore, setShowPointsStore] = useState(false);
  const [stockInfo, setStockInfo] = useState({});
  const [userInfo, setUserInfo] = useState({});
  const [showGameEndModal, setShowGameEndModal] = useState(false);

  const handleClosePointsStore = () => setShowPointsStore(false);
  const handleShowPointsStore = () => setShowPointsStore(true);

  Modal.setAppElement(rootElement);

  useEffect(() => {
    const randomDate = new Date(2023, 0, 1 + Math.floor(Math.random() * 300)); // 随机生成 2023-01-01 到 2023-12-01 之间的日期
    console.log("Date::", randomDate)
    const fetchTradingDay = async () => {
      try {
        const response = await axios.post('http://localhost:8000/api/next_trading_day', {
          current_date: randomDate.toISOString().split('T')[0],
          n: 1 // 获取当前日期的交易日
        });
        const tradingDay = new Date(response.data.next_trading_day);
        setStartDate(tradingDay);
        setCurrentDate(tradingDay);
      } catch (error) {
        console.error('Error fetching next trading day:', error);
      }
    };

    fetchTradingDay();
  }, []); // 仅在组件首次加载时调用一次

  useEffect(() => {
    setModelList([difficulty]);
  }, [difficulty]);

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

    if (selectedStock && currentDate) { // 确保在 currentDate 已设置之后再调用
      fetchStockData();
    }
  }, [selectedStock, currentDate]);

  useEffect(() => {
    console.log("StockData changed:", stockData);
  }, [stockData]);

  useEffect(() => {
    if (!startDate) return; // 确保 startDate 已设置之后再初始化游戏信息
    const initializeGameInfo = async () => {
      try {
        const initialData = {
          game_id: gameId,
          user_id: userId,
          cash: initialBalance,
          portfolio_value: 0,
          total_assets: initialBalance,
          stocks: selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: 0 }), {}),
          score: 0
        };
        setUserInfo(initialData);
        const aiData = {
          game_id: gameId,
          user_id: 'ai',
          cash: initialBalance,
          portfolio_value: 0,
          total_assets: initialBalance,
          stocks: {},
          score: 0
        };
        await axios.post('http://localhost:8000/api/game_info', initialData);
        await axios.post('http://localhost:8000/api/game_info', aiData);

        // 初始化交易记录
        const initialTransaction = {
          game_id: gameId,
          user_id: 'ai',
          stock_symbol: 'INIT',
          transaction_type: 'init',
          amount: 0,
          price: 0,
          date: startDate.toISOString()
        };
        await saveTransaction(initialTransaction);

        // 立即运行AI策略，确保有初始交易记录
        await runAIStrategy();
      } catch (error) {
        console.error('Error initializing game info:', error);
      }
    };

    initializeGameInfo();
    fetchStockInfo();
    console.log('initialize');
  }, [startDate]);

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
  }, [counter, gameEnd, stopCounter]);

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
      setStopCounter(false);
      closeModal();
    } else
      return;

    try {
      const response = await axios.post('http://localhost:8000/api/run_strategy', {
        tickers: selectedTickers,
        game_id: gameId,
        start_test_date: currentDate
      });
      console.log('Strategy result:', response.data);

      // 保存结果到数据库
      for (const record of response.data.trade_log) {
        const logEntry = {
          ...record,
          model: modelList[0],
          game_id: gameId
        };
        await axios.post('http://localhost:8000/api/save_trade_log', logEntry);
      }

    } catch (error) {
      console.error('Error running strategy:', error);
    }
  };

  const saveTransaction = async (transaction) => {
    try {
      await axios.post('http://localhost:8000/api/transactions', transaction);
      console.log('Transaction saved:', transaction);
    } catch (error) {
      console.error('Error saving transaction:', error);
    }
  };

  const runAIStrategy = async () => {
    const date = currentDate.toISOString().split('T')[0];
    const aiInfo = await fetchAiInfo();

    try {
      const aiResponse = await axios.get('http://localhost:8000/api/get_trade_log', {
        params: {
          game_id: gameId,
          model: modelList[0],
          date: date,
        }
      });

      if (aiResponse.data) {
        console.log("AI Strategy:", aiResponse.data);

        // 直接使用从后端获取的ai策略数据
        const strategy = aiResponse.data.change || {};
        if (Object.keys(strategy).length === 0) {
          selectedStockList.forEach(stock => {
            strategy[stock] = 0;
          });
        }

        for (const [stock, amount] of Object.entries(strategy)) {
          const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
            params: {
              symbol: stock,
              start_date: date,
              end_date: date
            }
          });
          const filteredData = response.data;

          if (filteredData.length === 0) {
            console.error(`Stock info for ${stock} on ${date} not found`);
            continue;
          }

          const stockInfo = filteredData[0];
          console.log("AI stockInfo:", stockInfo);

          const aiTransaction = {
            game_id: gameId,
            user_id: 'ai',
            stock_symbol: stock,
            transaction_type: amount > 0 ? 'buy' : (amount === 0 ? 'hold' : 'sell'),
            amount: Math.abs(amount),
            price: stockInfo.open,
            date: currentDate.toISOString()
          };

          await saveTransaction(aiTransaction);

          if (amount > 0) {
            aiInfo.cash -= stockInfo.open * amount;
            aiInfo.stocks[stock] = (aiInfo.stocks[stock] || 0) + amount;
          } else {
            aiInfo.cash += stockInfo.open * Math.abs(amount);
            aiInfo.stocks[stock] = (aiInfo.stocks[stock] || 0) - Math.abs(amount);
          }
        }

        const aiPortfolioValue = await selectedStockList.reduce(async (accPromise, stock) => {
          const acc = await accPromise;
          const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
            params: {
              symbol: stock,
              start_date: date,
              end_date: date
            }
          });
          const filteredData = response.data;

          if (filteredData.length === 0) {
            console.error(`Stock info for ${stock} on ${date} not found`);
            return acc;
          }

          const stockInfo = filteredData[0];
          return acc + (aiInfo.stocks[stock] || 0) * stockInfo.close;
        }, Promise.resolve(0));

        aiInfo.portfolio_value = aiPortfolioValue;
        aiInfo.total_assets = aiInfo.cash + aiInfo.portfolio_value;

        try {
          await axios.post('http://localhost:8000/api/game_info', aiInfo);
        } catch (error) {
          console.error('Error updating AI info:', error);
        }

        setAiStrategy(aiResponse.data); // 更新状态以触发其他依赖此状态的UI变化
        setShowStrategyModal(true);

        // 更新AI的状态
        setAiCash(aiInfo.cash);
        setAiPortfolioValue(aiInfo.portfolio_value);
        setAiTotalAssets(aiInfo.total_assets);
      } else {
        console.error('No AI strategy found');
      }
    } catch (error) {
      console.error('Error fetching AI strategy:', error);
    }
  };
  const fetchStockInfo = async () => {
    const date = currentDate.toISOString().split('T')[0]; // 确保date是一个字符串
    let userInfo2 = await fetchUserInfo();
    console.log("fetchStockInfo userInfo2:", userInfo2);

    if (!userInfo2) {
      userInfo2 = userInfo
      console.log("fetchUserinfo failed");
      return;
    }

    const newStockInfo = {};

    for (const stock of Object.keys(selectedTrades)) {
      console.log("useEffect stock:", stock);
      try {
        const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
          params: {
            symbol: stock,
            start_date: date,
            end_date: date
          }
        });

        if (response.data && response.data[0]) {
          newStockInfo[stock] = response.data[0];
        } else {
          console.error(`Stock info for ${stock} on ${date} not found`);
        }
      } catch (error) {
        console.error(`Error fetching stock data for ${stock}:`, error);
      }
    }

    setStockInfo(newStockInfo);
    console.log("fetchStockInfo stockInfo:", newStockInfo);

    if (!userInfo2.stocks) {
      userInfo2.stocks = selectedStockList.reduce((acc, stock) => ({ ...acc, [stock]: 0 }), {});
    }

    const portfolioValue = await selectedStockList.reduce(async (accPromise, stock) => {
      const acc = await accPromise;
      console.log("stock", stock);
      console.log("date", date);
      const response = await axios.get('http://localhost:8000/api/stored_stock_data', {
        params: {
          symbol: stock,
          start_date: date,
          end_date: date
        }
      });
      const filteredData = response.data;

      if (filteredData.length === 0) {
        console.error(`Stock info for ${stock} on ${date} not found`);
        return acc;
      }

      const stockInfo = filteredData[0];
      console.log("stockInfo:", stockInfo);
      return acc + (userInfo2.stocks[stock] || 0) * stockInfo.close;
    }, Promise.resolve(0));

    userInfo2.portfolio_value = portfolioValue;
    userInfo2.total_assets = userInfo2.cash + userInfo2.portfolio_value;

    // 更新前端显示的余额值
    setCash(userInfo2.cash);
    setPortfolioValue(userInfo2.portfolio_value);
    setTotalAssets(userInfo2.total_assets);
    setUserInfo(userInfo2);
    console.log('userinfo:', userInfo2);

    console.log("fetchStockInfo userInfo:", userInfo2);
  };

  useEffect(() => {
    if (currentDate) {
      console.log("useEffect currentDate:", currentDate);
      fetchStockInfo();
    }
  }, [currentDate, selectedTrades]);



  const handleSubmit = async () => {
    console.log('handleSubmit:');
    const userInfo2 = await fetchUserInfo();
    for (const [stock, { type, amount }] of Object.entries(selectedTrades)) {
      const transaction = {
        game_id: gameId,
        user_id: userId,
        stock_symbol: stock,
        transaction_type: type,
        amount: parseFloat(amount),
        price: stockInfo[stock].open,
        date: currentDate.toISOString()
      };

      await saveTransaction(transaction);

      if (type === 'buy') {
        userInfo2.cash -= stockInfo[stock].open * amount;
        userInfo2.stocks[stock] = (userInfo2.stocks[stock] || 0) + parseFloat(amount);
      } else if (type === 'sell') {
        userInfo2.cash += stockInfo[stock].open * amount;
        userInfo2.stocks[stock] = (userInfo2.stocks[stock] || 0) - Math.abs(amount);
      }
    }

    console.log('submit userInfo2:', userInfo2);

    // 先保存用户的game_info
    try {
      await axios.post('http://localhost:8000/api/game_info', userInfo2);
    } catch (error) {
      console.error('Error updating user info:', error);
    }

    // 保存AI的交易记录
    await runAIStrategy();

    setStopCounter(true);

  };

  const fetchUserInfo = async () => {
    try {
      console.log("fetchUserInfo gameId:", gameId);
      console.log("fetchUserInfo userId:", userId);
      const response = await axios.get('http://localhost:8000/api/game_info', {
        params: {
          game_id: gameId,
          user_id: userId
        }
      });
      console.log("fetchUserInfo response:", response);
      return response.data[0]; // 假设返回的第一个是需要的用户信息
    } catch (error) {
      console.error('Error fetching user info:', error);
      return null;
    }
  };

  const fetchAiInfo = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/game_info', {
        params: {
          game_id: gameId,
          user_id: 'ai' // 假设AI用户ID为 'ai'
        }
      });
      return response.data[0]; // 假设返回的第一个是需要的AI信息
    } catch (error) {
      console.error('Error fetching AI info:', error);
      return null;
    }
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
        setShowGameEndModal(true); // 显示游戏结束的模态框
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

  if (!currentDate) {
    return <div>Loading...</div>; // 在 currentDate 未设置之前显示加载状态
  }

  return (
    <div className="background">
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100" style={{ color: 'white' }}>
          <AppHeader />
          <div className="d-flex justify-content-between align-items-center w-100">
            <div className="d-flex justify-content-start" style={{ padding: '1em' }}>
              AI Opponent: {difficulty}
            </div>
            <div className="d-flex justify-content-center align-items-center flex-grow-1">
              <span className="mx-3">Current Round: {currentRound}/{MaxRound}</span>
              <span className="mx-3">Current Date: {currentDate.toISOString().split('T')[0]}</span>
              <span className="mx-3">Countdown: {counter}</span>
            </div>
            <CDropdown variant="dropdown">
              <CDropdownToggle caret={true}>
                <span style={{ color: 'white' }}>Game Credits: 50</span>
              </CDropdownToggle>

              <CDropdownMenu className='dropdown-menu'>
                <CDropdownItem className='dropdown-item' onClick={handleShowPointsStore}>
                  <span style={{ color: 'white' }}>Shop</span>
                </CDropdownItem>
              </CDropdownMenu>
            </CDropdown>

            {/* Points Store Modal */}
            <PointsStoreModal show={showPointsStore} handleClose={handleClosePointsStore} />
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
            <div className="bottom-section d-flex">
              <div className="left-section" style={{ flex: 1, marginRight: '20px' }}>
                <StockTradeComponent
                  selectedTrades={selectedTrades}
                  setSelectedTrades={setSelectedTrades}
                  initialBalance={initialBalance}
                  cash={cash}
                  userId={userId}
                  selectedStock={selectedStockList}
                  handleSubmit={handleSubmit}
                  stockData={stockInfo}
                  userInfo={userInfo}
                />
              </div>
              <div className="right-section" style={{ flex: 1 }}>
                <div className="financials mb-3">
                  <div className="d-flex justify-content-between align-items-center w-100 mb-3">
                    <div style={{ marginTop: '21px', marginRight: '4px' }}>Cash: ${cash.toFixed(2)}</div>
                    <div style={{ marginTop: '21px', marginRight: '4px' }}>Portfolio Value: ${portfolioValue.toFixed(2)}</div>
                    <div style={{ marginTop: '21px', marginRight: '4px' }}>Total Assets: ${totalAssets.toFixed(2)}</div>
                  </div>
                  <div className="d-flex justify-content-between align-items-center w-100 mb-3">
                    <div style={{ marginRight: '4px' }}>AI Cash: ${aiCash.toFixed(2)}</div>
                    <div style={{ marginRight: '4px' }}>AI Portfolio Value: ${aiPortfolioValue.toFixed(2)}</div>
                    <div style={{ marginRight: '4px' }}>AI Total Assets: ${aiTotalAssets.toFixed(2)}</div>
                  </div>
                </div>
                <Standings initialBalance={initialBalance} totalAssets={totalAssets} aiTotalAssets={aiTotalAssets} />
              </div>
            </div>



            <div className="history-section">
              <TradeHistory userId={userId} refreshHistory={refreshHistory} selectedStock={selectedStockList} gameId={gameId} />
            </div>
          </div>
        </div>
      </div>

      <Modal isOpen={isModalOpen} onRequestClose={closeModal} contentLabel="Select Stocks"
        shouldCloseOnEsc={false}
        shouldCloseOnOverlayClick={false}
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
        <h2 style={{ textAlign: 'center', color: '#333', marginBottom: '20px' }}>Please Select 3 Stocks</h2>
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
          <Button
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
          </Button>
        </div>
      </Modal>


      <Modal
        isOpen={showStrategyModal}
        onRequestClose={closeStrategyModal}
        contentLabel="Strategy Modal"
        style={{
            overlay: {
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(5px)',
            zIndex: '1000'
            },
            content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '80%',
            maxWidth: '600px',
            height: 'auto',
            padding: '20px',
            borderRadius: '10px',
            border: 'none',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
            backgroundColor: '#fff',
            color: '#333',
            textAlign: 'center' // 新增的样式
            }
        }}
        >
        <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>
            Strategy for {currentDate.toISOString().split('T')[0]}
        </h2>
        <div style={{ marginBottom: '20px' }}>
            <h3 style={{ marginBottom: '10px', color: '#444' }}>Player's Strategy:</h3>
            {Object.entries(selectedTrades).map(([stock, trade]) => (
            <div key={stock} style={{ marginBottom: '10px' }}>
                <p>{stock}: {trade.type} {trade.amount}</p>
            </div>
            ))}
        </div>
        <div style={{ marginBottom: '20px' }}>
            <h3 style={{ marginBottom: '10px', color: '#444' }}>AI's Strategy:</h3>
            {aiStrategy && aiStrategy.change ? (
            Object.entries(aiStrategy.change).map(([stock, change]) => (
                <div key={stock} style={{ marginBottom: '10px' }}>
                <p>{stock}: {change > 0 ? `buy ${change}` : (change === 0 ? `hold ${change}` : `sell ${Math.abs(change)}`)}</p>
                </div>
            ))
            ) : (
            <p>No AI strategy found</p>
            )}
        </div>
        <Button
            onClick={closeStrategyModal}
            style={{
            display: 'block',
            margin: '0 auto',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '5px',
            backgroundColor: '#007bff',
            color: '#fff',
            cursor: 'pointer'
            }}
            variant='outlined'
        >
            Close
        </Button>
      </Modal>
      {/* Game End Modal */}
      <GameEndModal
        isOpen={showGameEndModal}
        onRequestClose={() => setShowGameEndModal(false)}
        userAssets={totalAssets}
        aiAssets={aiTotalAssets}
        userProfit={totalAssets - initialBalance}
        aiProfit={aiTotalAssets - initialBalance}
      />
    </div >
  );
}

export default CompetitionLayout;
