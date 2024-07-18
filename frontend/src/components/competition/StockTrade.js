import React, { useState, useEffect } from 'react';
import Button from '@mui/material/Button';
import { Checkbox } from '@mui/material';

const StockTradeComponent = ({ selectedTrades, setSelectedTrades, initialBalance, cash, userId, selectedStock, handleSubmit, stockData, userInfo }) => {
  // const [tempCash, setTempCash] = useState(cash);
  const [gameEnd, setGameEnd] = useState(false);
  console.log('stockData:', stockData);

  useEffect(() => {
    // 初始化默认选择为hold
    const initialTrades = {};
    selectedStock.forEach(stock => {
      initialTrades[stock] = { type: 'hold', amount: '0' };
    });
    setSelectedTrades(initialTrades);
  }, [selectedStock]);

  console.log('selectedTrades:', selectedTrades);

  const handleBuySellChange = (stock, type, amount) => {
    setSelectedTrades((prevTrades) => ({
      ...prevTrades,
      [stock]: { type, amount }
    }));
  };

  const handleHoldChange = (stock) => {
    setSelectedTrades((prevTrades) => ({
      ...prevTrades,
      [stock]: { type: 'hold', amount: '0' }
    }));
  };

  const handleClear = () => {
    const clearedTrades = {};
    selectedStock.forEach(stock => {
      clearedTrades[stock] = { type: 'hold', amount: '0' };
    });
    setSelectedTrades(clearedTrades);
  };

  const onBuyInput = (e, stock) => {
    let value = Math.floor(e.target.value);
    value = Math.max(value, 0);
    console.log('onBuyInput stockData:', stockData);
    let _cash = cash;
    for (let stockOther in stockData)
      if (stockOther !== stock)
        if (selectedTrades[stockOther].type === 'buy')
          _cash -= stockData[stockOther].open * selectedTrades[stockOther].amount;
        else if (selectedTrades[stockOther].type === 'sell')
          _cash += stockData[stockOther].open * selectedTrades[stockOther].amount;
    console.log('onBuyInput _cash:', _cash);
    value = Math.min(value, Math.floor(_cash / stockData[stock].open));
    e.target.value = value;
    return value;
  }

  const onSellInput = (e, stock) => {
    console.log('onSellInput stockData:', userInfo);
    let value = Math.floor(e.target.value);
    value = Math.max(value, 0);
    value = Math.min(value, userInfo['stocks'][stock]);
    e.target.value = value;
    return value;
  }

  const getStockAmount = (stock) => {
    console.log('getStockAmount userInfo:', userInfo);
    if (userInfo && userInfo['stocks'] && userInfo['stocks'][stock] !== undefined) {
      return userInfo['stocks'][stock];
    } else {
      return 0; // 默认为0股份
    }
  };

  return (
    <div className="decision-area">
      <div className="stock-container">
        {selectedStock.map((stock) => (
          <div key={stock} className="stock-item" style={{ flex: '1' }}>
            <div className="stock-name">{stock} Share: {getStockAmount(stock)}</div>
            <div className="trade-options">
              <div>
                <Checkbox
                  type="radio"
                  name={`${stock}-trade`}
                  value="buy"
                  onChange={(e) => handleBuySellChange(stock, 'buy', '0')}
                  checked={selectedTrades[stock]?.type === 'buy'}
                />
                Buy &emsp;
                {selectedTrades[stock]?.type === 'buy' && (
                  <input
                    type="number"
                    value={selectedTrades[stock]?.amount || ''}
                    onChange={(e) => handleBuySellChange(stock, 'buy', e.target.value)}
                    style={{ minWidth: '0', width: '90%' }}
                    min={0}
                    step={1}
                    onInput={(e) => onBuyInput(e, stock)}
                  />
                )}
              </div>
              <div>
                <Checkbox
                  type="radio"
                  name={`${stock}-trade`}
                  value="sell"
                  onChange={(e) => handleBuySellChange(stock, 'sell', '0')}
                  checked={selectedTrades[stock]?.type === 'sell'}
                />
                Sell &emsp;
                {selectedTrades[stock]?.type === 'sell' && (
                  <input
                    type="number"
                    value={selectedTrades[stock]?.amount || ''}
                    onChange={(e) => handleBuySellChange(stock, 'sell', e.target.value)}
                    style={{ minWidth: '0', width: '90%' }}
                    min={0}
                    step={1}
                    onInput={(e) => onSellInput(e, stock)}
                  />
                )}
              </div>
              <div>
                <Checkbox
                  type="radio"
                  name={`${stock}-trade`}
                  value="hold"
                  onChange={() => handleHoldChange(stock)}
                  checked={selectedTrades[stock]?.type === 'hold'}
                />
                Hold &emsp;
              </div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
        <Button className="clear-button" onClick={handleClear} variant='outlined' style={{ color: '#e3f2fd' }}>Clear</Button>
        <Button
          className={`submit-button ${Object.values(selectedTrades).length > 0 && !gameEnd ? 'active' : 'disabled'}`}
          variant='outlined'
          style={{ color: '#e0f2f1' }}
          onClick={handleSubmit}
          disabled={Object.keys(selectedTrades).length === 0}
        >
          Submit
        </Button>
      </div>
    </div>
  );
};

export default StockTradeComponent;
