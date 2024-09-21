# RLTrader - Reinforcement Learning for Stock Trading                                                         

![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-7B0099?style=for-the-badge&logo=yahoo&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Overview

This project implements an AI-based stock trading system using Reinforcement Learning (RL) and LSTM Neural Networks. It utilizes historical stock data from [Yahoo Finance](https://finance.yahoo.com) to train an agent capable of buying, holding, or selling stocks based on stock prices. The primary goal of this agent is to maximize profit by making strategic decisions based on past data patterns.

The project is built using **TensorFlow** to define and train the neural network and uses **Q-Learning** as the reinforcement learning algorithm. The model uses multiple LSTM layers to capture the temporal dependencies in stock prices, which helps the AI make more informed trading decisions.

## Features

- **Reinforcement Learning**: Utilizes Q-Learning to reward the agent based on actions taken (buy/sell/hold) and outcomes (profit/loss).
- **LSTM Neural Network**: Leveraging LSTM layers to analyze time-series stock data, which helps the agent to predict future price movements.
- **Experience Replay**: Storing past experiences in a memory buffer, allowing the agent to learn from previous actions and improve its decision-making process.
- **Yahoo Finance API**: The stock data used for training is pulled directly from Yahoo Finance using the `yfinance` Python package.

## How It Works

1. **Data Acquisition**: Stock price data is downloaded using the Yahoo Finance API.
2. **State Representation**: A sliding window of historical prices is used to represent the current state of the stock for the agent.
3. **Actions**: The agent can:
    - Buy a stock
    - Hold a stock (do nothing)
    - Sell a stock
4. **Rewards**: After taking an action, the agent receives a reward based on the profit or loss from selling stocks.
5. **Training**: The agent is trained over multiple episodes, adjusting its strategy by improving its policy through Q-Learning.
6. **Performance Monitoring**: The system tracks total profit after each episode and saves the trained model for future use.

## Tools & Libraries

- **TensorFlow**: Used for constructing and training the LSTM-based neural network model.
- **Yahoo Finance API**: Used for retrieving historical stock price data for training the model.
- **NumPy**: For data manipulation and numerical operations.
- **Pandas**: For loading and preprocessing stock price data.
- **Matplotlib**: For visualizing stock prices and performance.
- **tqdm**: For monitoring training progress with progress bars.

## Results

The AI Trader was tested on real historical data and exhibited promising performance, achieving profitability in several trading episodes. The model showed improvement as training progressed, reducing losses and capitalizing on stock price movements.

## Getting Started

### Prerequisites

Before you can run this project, you need to have the following installed:

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Yahoo Finance API (`yfinance`)
- tqdm

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AI_Stock_Trader.git
   cd AI_Stock_Trader
   ```
   
## Install the required packages

```bash
pip install -r requirements.txt
```

### Usage
You can customize the stock ticker, trading parameters, and training episodes in the script to test the AI Trader on different stocks or trading strategies.

### Future Improvements
- Risk Management: Introduce risk management techniques like stop-loss and profit targets.
- Hyperparameter Tuning: Optimize the model by experimenting with different neural network architectures and reinforcement learning parameters.
- Real-time Trading: Implement a real-time trading system that can trade stocks in live markets, using tools such as Kafka and Zookeeper.
  
### Acknowledgements
The historical stock data is sourced from Yahoo Finance. This project uses TensorFlow to implement and train the neural network.

