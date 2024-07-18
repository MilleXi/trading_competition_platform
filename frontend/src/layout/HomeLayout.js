import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../css/home.css';
import {
  AppBreadcrumb,
  AppContent,
  AppFooter,
  AppHeader,
  AppHeaderDropdown,
  AppSidebar,
  DocsCallout,
  DocsLink,
  DocsExample,
} from '../components/index';
import {
    DifficultyModal,
    GameButtons,
    GameIntro,
    GameLogo,
} from '../components/home/index';

const HomeLayout = () => {
  const [difficultyModalVisible, setDifficultyModalVisible] = useState(false);
  const [difficulty, setDifficulty] = useState('LSTM');
  const navigate = useNavigate();

  const startGame = () => {
    navigate('/competition', { state: { difficulty } });
    console.log("Competition starts")
  }

  // 将状态提升到父组件，以便在点击按钮时更新状态
  const handleDifficultyClick = (difficulty) => {
    setDifficulty(difficulty);
    setDifficultyModalVisible(false);
  };

  return (
    <div className="background">
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100" style={{ color: 'white' }}>
          <AppHeader />
          <div className="body flex-grow-1 px-3 d-flex flex-column align-items-center">
            <GameLogo />
            <GameIntro />
            <GameButtons onStartClick={startGame} onDifficultyClick={() => setDifficultyModalVisible(true)} difficulty={difficulty} />
          </div>
        </div>
      </div>
      <DifficultyModal visible={difficultyModalVisible} onClose={() => setDifficultyModalVisible(false)} onDifficultyClick={handleDifficultyClick} />
    </div>
  );
}

export default HomeLayout;
