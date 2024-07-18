import React from 'react';
import { CModal, CModalHeader, CModalBody, CButton, CModalTitle } from '@coreui/react';

const DifficultyModal = ({ visible, onClose, onDifficultyClick }) => {
  return (
    <CModal visible={visible} onClose={onClose} className="custom-modal">
      <CModalHeader onClose={onClose}>
        <CModalTitle>AI Opponent Selection</CModalTitle>
      </CModalHeader>
      <CModalBody>
        <div className="difficulty-options">
          <CButton color="primary" className="mb-2" onClick={() => onDifficultyClick('LSTM')}>LSTM</CButton>
          <CButton color="warning" className="mb-2" onClick={() => onDifficultyClick('LGBM')}>LGBM</CButton>
          <CButton color='danger' className="mb-2" onClick={() => onDifficultyClick('XGBoost')}>XGBoost</CButton>
        </div>
      </CModalBody>
    </CModal>
  );
}

export default DifficultyModal;
