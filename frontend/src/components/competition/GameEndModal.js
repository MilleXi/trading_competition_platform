import React from 'react';
import Modal from 'react-modal';
import { useNavigate } from 'react-router-dom';

const GameEndModal = ({ isOpen, onRequestClose, userAssets, aiAssets, userProfit, aiProfit }) => {
  const navigate = useNavigate();
  const handleHomeRedirect = () => {
    onRequestClose();
    navigate('/');
  };

  const isUserWinner = userAssets > aiAssets;

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onRequestClose}
      contentLabel="Game End Modal"
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
          color: 'black',
        }
      }}
    >
      <h2>Game Over</h2>
      <p>{isUserWinner ? 'Congratulations! You won!' : 'You lost. Better luck next time!'}</p>
      <div>
        <h3>Final Results:</h3>
        <p>Player's Total Assets: ${userAssets.toFixed(2)}</p>
        <p>AI's Total Assets: ${aiAssets.toFixed(2)}</p>
        <p>Player's Total Profit: ${userProfit.toFixed(2)}</p>
        <p>AI's Total Profit: ${aiProfit.toFixed(2)}</p>
      </div>
      <button onClick={handleHomeRedirect} style={{ display: 'block', margin: '20px auto' }}>Home</button>
    </Modal>
  );
};

export default GameEndModal;
