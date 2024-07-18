import React, { useState, useEffect } from 'react';
import { Modal, Button, FormControlLabel, Checkbox, TextField, IconButton } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';

const pointsStoreItems = [
  {
    label: 'Intelligence Reduction Pill',
    description: "Permanently reduces the AI's intelligence level.",
    price: 200,
  },
  {
    label: 'Half Intelligence Reduction Pill',
    description: "Temporarily reduces the AI's intelligence level for one round.",
    price: 100,
  },
  {
    label: 'Signal Interference',
    description: 'Prevents the AI from executing trades for one round.',
    price: 200,
  },
  {
    label: 'Steal and Replace',
    description:
      "Transfers a portion of the AI's balance to your account. Price: 300 points per purchase, with each purchase transferring $100. This item can be purchased multiple times.",
    price: 300,
  },
  {
    label: 'Points Cash Out',
    description: 'Converts points into account balance. Exchange rate: 5 points = $3, with a minimum redemption of 500 points.',
    price: 0,
  },
];

const PointsStoreModal = ({ show, handleClose }) => {
  const [selectedItems, setSelectedItems] = useState({});
  const [quantity, setQuantity] = useState(1);
  const [pointsAmount, setPointsAmount] = useState(500);
  const [totalPoints, setTotalPoints] = useState(0);

  const handleCheckboxChange = (item) => {
    setSelectedItems((prev) => ({
      ...prev,
      [item.label]: !prev[item.label],
    }));
  };

  useEffect(() => {
    let total = 0;
    for (const item of pointsStoreItems) {
      if (selectedItems[item.label]) {
        if (item.label === 'Steal and Replace') {
          total += item.price * quantity;
        } else if (item.label === 'Points Cash Out') {
          total += pointsAmount >= 500 ? pointsAmount : 0;
        } else {
          total += item.price;
        }
      }
    }
    setTotalPoints(total);
  }, [selectedItems, quantity, pointsAmount]);

  const handlePurchase = () => {
    const selected = Object.keys(selectedItems).filter((key) => selectedItems[key]);
    console.log('Selected items for purchase:', selected);
    // Add your purchase logic here
  };

  return (
    <Modal open={show} onClose={handleClose}>
      <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px' }}>
        <h2 style={{ color: 'black' }}>Points Store</h2>
        <ul style={{ color: 'black' }}>
          {pointsStoreItems.map((item) => (
            <li key={item.label}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={!!selectedItems[item.label]}
                    onChange={() => handleCheckboxChange(item)}
                  />
                }
                label={
                  <div style={{ color: 'black' }}>
                    <strong>{item.label}:</strong> {item.description} Price: {item.price} points
                  </div>
                }
              />
              {item.label === 'Steal and Replace' && selectedItems[item.label] && (
                <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px' }}>
                  <IconButton onClick={() => setQuantity(Math.max(1, quantity - 1))}>
                    <RemoveIcon />
                  </IconButton>
                  <TextField
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(Math.max(1, parseInt(e.target.value) || 1))}
                    style={{ width: '60px', margin: '0 10px' }}
                  />
                  <IconButton onClick={() => setQuantity(quantity + 1)}>
                    <AddIcon />
                  </IconButton>
                  <span style={{ marginLeft: '10px' }}>Total Points: {quantity * item.price}</span>
                </div>
              )}
              {item.label === 'Points Cash Out' && selectedItems[item.label] && (
                <div style={{ marginTop: '10px' }}>
                  <TextField
                    type="number"
                    label="Points to Exchange"
                    value={pointsAmount}
                    onChange={(e) => setPointsAmount(Math.max(0, parseInt(e.target.value) || 0))}
                    fullWidth
                  />
                  <span style={{ color: 'red', display: pointsAmount < 500 ? 'block' : 'none' }}>
                    Minimum 500 points required
                  </span>
                </div>
              )}
            </li>
          ))}
        </ul>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px' }}>
          <div style={{ color: 'black', fontWeight: 'bold' }}>
            Total Points: {totalPoints}
          </div>
          <div>
            <Button variant="contained" color="secondary" onClick={handleClose} style={{ marginRight: '10px' }}>
              Close
            </Button>
            <Button
              variant="contained"
              color="primary"
              onClick={handlePurchase}
              disabled={selectedItems['Points Cash Out'] && pointsAmount < 500}
            >
              Purchase
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  );
};

export default PointsStoreModal;
