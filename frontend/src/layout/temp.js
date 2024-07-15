import React, { useState } from 'react';
import '../css/index.css';
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
} from '../components/index'

import { CModal, CModalHeader, CModalBody, CModalFooter, CButton, CModalTitle } from '@coreui/react';

const GameLayout = () => {
    const [difficultyModalVisible, setDifficultyModalVisible] = useState(false);

    const toggleDifficultyModal = () => {
      setDifficultyModalVisible(!difficultyModalVisible);
    };

  return (
    <div className="background"> 
      <div className="app">
        <div className="wrapper d-flex flex-column min-vh-100">
          <AppHeader>
            <AppHeaderDropdown />
          </AppHeader>
          <div className="body flex-grow-1 px-3 d-flex flex-column align-items-center">
            {/* 添加游戏标志 */}
            <img src="/src/assets/images/logo.png" alt="Game Logo" className="game-logo" />
            {/* 添加项目标题和简要介绍 */}
            <div className="game-intro text-center">
              <h1 className="game-title">股票对战</h1>
              <p className="game-description">
                欢迎来到我们的股票对战游戏！在这里，你可以与AI进行股票交易对决，展示你的投资策略和技巧，看看你能否战胜AI，成为股市赢家！
              </p>
            </div>
            {/* 添加开始游戏按钮和难度选择按钮 */}
            <div className="game-buttons">
              <button className="btn btn-primary start-game-button">开始游戏</button>
              <button className="btn btn-secondary difficulty-button" onClick={toggleDifficultyModal}>选择难度</button>
            </div>
          </div>
        </div>
        <AppFooter />
      </div>
      {/* 难度选择框 */}
      <CModal visible={difficultyModalVisible} onClose={toggleDifficultyModal}>
        <CModalHeader onClose={toggleDifficultyModal}>
          <CModalTitle>选择难度</CModalTitle>
        </CModalHeader>
        <CModalBody>
          <div className="difficulty-options">
            <CButton color="primary" className="mb-2">简单</CButton>
            <CButton color="warning" className="mb-2">中等</CButton>
            <CButton color="danger" className="mb-2">困难</CButton>
          </div>
        </CModalBody>
      </CModal>
    </div>
  )
}

export default GameLayout;