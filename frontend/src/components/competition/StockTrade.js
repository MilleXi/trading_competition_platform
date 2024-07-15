import React, { useState } from 'react';

const StockTradeComponent = ({ initialBalance }) => {
  const [selectedTrades, setSelectedTrades] = useState({
    stock1: { type: 'hold', amount: '' },
    stock2: { type: 'hold', amount: '' },
    stock3: { type: 'hold', amount: '' }
  });
  const [balance, setBalance] = useState(initialBalance);
  const [remainingBalance, setRemainingBalance] = useState(initialBalance);
  const [gameEnd, setGameEnd] = useState(false);

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
      [stock]: { type: 'hold', amount: '' }
    }));
  };

  const handleClear = () => {
    setSelectedTrades({});
  };

  const handleSubmit = () => {
    // 提交逻辑，例如更新余额和股票状态
    console.log('Submitted trades:', selectedTrades);
  };

  return (
    <div className="decision-area">
      <div>Initial Investment: {balance}</div>
      <div>Balance: {remainingBalance}</div>

      <div className="stock-container">
        {['stock1', 'stock2', 'stock3'].map((stock) => (
          <div key={stock} className="stock-item" style={{flex: '1'}}>
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
                    style={{minWidth: '0', width: '90%'}}
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
                    style={{minWidth: '0', width: '90%'}}
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

      <button className="clear-button" onClick={handleClear}>Clear</button>
      <button
        className={`submit-button ${Object.values(selectedTrades).length > 0 && !gameEnd ? 'active' : 'disabled'}`}
        onClick={handleSubmit}
        disabled={Object.keys(selectedTrades).length === 0}
      >
        Submit
      </button>
    </div>
  );
};

export default StockTradeComponent;
