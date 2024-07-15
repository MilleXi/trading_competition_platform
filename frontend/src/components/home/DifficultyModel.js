import React from 'react';
import { CModal, CModalHeader, CModalBody, CButton, CModalTitle } from '@coreui/react';

const DifficultyModal = ({ visible, onClose, onDifficultyClick }) => {
  return (
    <CModal visible={visible} onClose={onClose} className="custom-modal">
      <CModalHeader onClose={onClose}>
        <CModalTitle>Difficulty Selection</CModalTitle>
      </CModalHeader>
      <CModalBody>
        <div className="difficulty-options">
          <CButton color="primary" className="mb-2" onClick={() => onDifficultyClick('Easy')}>Easy</CButton>
          <CButton color="warning" className="mb-2" onClick={() => onDifficultyClick('Medium')}>Medium</CButton>
          <CButton color='danger' className="mb-2" onClick={() => onDifficultyClick('Hard')}>Hard</CButton>
        </div>
      </CModalBody>
    </CModal>
  );
}

export default DifficultyModal;
