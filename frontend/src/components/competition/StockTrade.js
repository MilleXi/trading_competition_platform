import React, { useState, useEffect } from 'react';
import Button from '@mui/material/Button';

const StockTradeComponent = ({ selectedTrades, setSelectedTrades, initialBalance, userId, selectedStock, handleSubmit }) => {
  const [balance, setBalance] = useState(initialBalance);
  const [remainingBalance, setRemainingBalance] = useState(initialBalance);
  const [gameEnd, setGameEnd] = useState(false);

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

  return (
    <div className="decision-area">
      <div>Initial Investment: {balance}</div>
      <div>Balance: {remainingBalance}</div>

      <div className="stock-container">
        {selectedStock.map((stock) => (
          <div key={stock} className="stock-item" style={{ flex: '1' }}>
            <div className="stock-name">{stock}</div>
            <div className="trade-options">
              <div>
                <input
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
                    onInput={(e) => e.target.value = Math.floor(e.target.value)}
                  />
                )}
              </div>
              <div>
                <input
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
                    onInput={(e) => e.target.value = Math.floor(e.target.value)}
                  />
                )}
              </div>
              <div>
                <input
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
        <Button className="clear-button" onClick={handleClear} variant='outlined' style={{color:'#e3f2fd'}}>Clear</Button>
        <Button
          className={`submit-button ${Object.values(selectedTrades).length > 0 && !gameEnd ? 'active' : 'disabled'}`}
          variant='outlined'
          style={{color:'#e0f2f1'}}
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
