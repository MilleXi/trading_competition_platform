# Trading Wars: Human vs AI Stock Trading Game

## Project Overview

Welcome to Trading Wars, an innovative and engaging platform where artificial intelligence meets human ingenuity in a thrilling stock trading competition. Trading Wars is designed to provide users with an interactive and competitive environment to test and enhance their trading skills against advanced AI models. The platform is not just a game but a comprehensive simulation that combines real-world financial data, sophisticated AI algorithms, and user-friendly interfaces to create an unparalleled trading experience.

## Key Features

- **AI Opponents**: Choose from three sophisticated AI models: LSTM, XGBoost, and LightGBM, each employing unique trading strategies.
- **Starting Capital**: Begin with a capital of $100,000.
- **Stock Selection**: Select three stocks from a list of 30 to focus on throughout the game.
- **10 Rounds**: The game is structured into 10 rounds, each representing a single trading day.
- **Financial Reports and K-Line Charts**: Receive updated financial reports and K-line charts for your chosen stocks at the beginning of each round.
- **Trade Decisions**: Make buy, sell, or hold decisions within one minute per round.
- **Performance Metrics**: Track remaining cash, current portfolio value, and total assets, with standings updated based on cumulative income.
- **Real-Time Feedback**: Immediate feedback on trade outcomes and performance.
- **Trade History**: Log of all trades made by both the player and the AI.
- **Points System**: Earn or lose points based on performance, which can be used to purchase in-game items.

## Game Guide Vedio

Watch the demo video below to see how to play the game:

[Watch the video](https://youtu.be/E7QLd853hZ0)

## Game Tips

- Diversify your portfolio to spread risk.
- Analyze financial reports and K-line charts to make informed decisions.
- Utilize the one-minute decision period each round to strategize effectively.

## Technology Stack

- **Frontend**: ReactJS
- **Backend**: Flask
- **Database**: SQLite
- **Stock Data**: yfinance

## Getting Started

### Clone the Project

```bash
git clone https://github.com/MilleXi/trading_competition_platform.git
cd trading_competition_platform
```

### Setup Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Backend Server

```bash
cd backend
flask run --port 8000
```

### Run the Frontend Server

```bash
cd frontend
npm install
npm start
```

## Contributing

We welcome any form of contribution! If you have new ideas, find bugs, or have code improvements, please submit an Issue or a Pull Request.

## License

This project is open-sourced under the MIT License. For details, please see the LICENSE file.

This project contains a modified version of [coreui-free-react-admin-template](https://github.com/coreui/coreui-free-react-admin-template) ([MIT License](https://github.com/coreui/coreui-free-react-admin-template/blob/main/LICENSE)) - Copyright (c) 2024 creativeLabs Łukasz Holeczek.

## Acknowledgements

I would like to extend my heartfelt gratitude to my amazing teammates who have contributed significantly to this project. Working together has been an incredibly rewarding and enjoyable experience. Please feel free to follow them on GitHub to see their great work:

- [@Rigel-Alpha](https://github.com/Rigel-Alpha)😇
- [@yali-hzy](https://github.com/yali-hzy)
- [@Caromiobenn](https://github.com/Caromiobenn)

Their dedication and hard work have been instrumental in bringing this project to life. Thank you for your outstanding contributions and collaboration!

## Contact Us

If you have any questions or suggestions, please feel free to contact us:

- Email: xyy318@126.com
- GitHub Issues: [GitHub Issues](https://github.com/MilleXi/trading_competition_platform/issues)

Thank you for your participation and support!