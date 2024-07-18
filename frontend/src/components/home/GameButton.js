import React from 'react';

const GameButtons = ({onStartClick, onDifficultyClick, difficulty}) => {
  return (
    <div className="game-buttons">
      <button className="btn btn-primary start-game-button" onClick={onStartClick}>Start Game</button>
      <button className="btn btn-secondary difficulty-button" onClick={onDifficultyClick}>Model: {difficulty}</button>
    </div>
  );
}

export default GameButtons;
