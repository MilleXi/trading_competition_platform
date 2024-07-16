import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()
import os
import csv

# Define the classes and functions
class Deep_Evolution_Strategy:
    inputs = None

    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch=1000, print_every=1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print('iter %d. reward: %f' % (i + 1, self.reward_function(self.weights)))
        print('time taken to train:', time.time() - lasttime, 'seconds')


class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, window_size, trend, skip, initial_money, stock_name):
        self.model = model
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.initial_money = initial_money
        self.stock_name = stock_name
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

        # Initialize a list to store buy/sell records
        self.trade_records = []
        # Ensure the folder exists
        self.folder_name = 'selldata'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        # Create a CSV file and write the header row
        self.file_path = os.path.join(self.folder_name, f"{self.stock_name}_trade_records.csv")
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['day', 'operation', 'price', 'investment', 'total balance'])
            writer.writeheader()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0: t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def get_reward(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights
        state = self.get_state(0)
        inventory = []
        quantity = 0
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            if action == 1 and starting_money >= self.trend[t] * 10:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t] * 10

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t] * 10

            state = next_state
        return ((starting_money - initial_money) / initial_money) * 100

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        initial_money = self.initial_money
        state = self.get_state(0)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.trend[t] * 10:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t] * 10
                states_buy.append(t)
                print('day %d: buy 10 units at price %f, total balance %f' % (t, self.trend[t], initial_money))
                buy_record = {
                    'day': t,
                    'operation': 'buy',
                    'price': self.trend[t],
                    'investment': 0,
                    'total balance': initial_money
                }
                self.trade_records.append(buy_record)

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t] * 10
                states_sell.append(t)
                try:
                    invest = ((self.trend[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                sell_record = {
                    'day': t,
                    'operation': 'sell',
                    'price': self.trend[t],
                    'investment': invest,
                    'total balance': initial_money
                }
                self.trade_records.append(sell_record)
                print(
                    'day %d, sell 10 units at price %f, investment %f %%, total balance %f'
                    % (t, self.trend[t], invest, initial_money)
                )
            state = next_state

        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money

        # Append the trade records to the CSV file
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['day', 'operation', 'price', 'investment', 'total balance'])
            for record in self.trade_records:
                writer.writerow(record)

        return states_buy, states_sell, total_gains, invest


def process_stocks(data_folder='yeardata', output_folder='img_output'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_folder, file_name)
            df = pd.read_csv(file_path)
            stock_name = file_name.split('_')[0]
            close = df.Close.values.tolist()
            window_size = 30
            skip = 1
            initial_money = 100000

            model = Model(input_size=window_size, layer_size=500, output_size=3)
            agent = Agent(model=model,
                          window_size=window_size,
                          trend=close,
                          skip=skip,
                          initial_money=initial_money,
                          stock_name=stock_name)
            agent.fit(iterations=1000, checkpoint=10)
            states_buy, states_sell, total_gains, invest = agent.buy()
            fig = plt.figure(figsize=(15, 5))
            plt.plot(close, color='r', lw=2.)
            plt.plot(close, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
            plt.plot(close, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
            plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
            plt.legend()
            output_path = os.path.join(output_folder, f"{stock_name}_tradepic.png")
            plt.savefig(output_path)
            plt.close()


# Run the process for all stocks in the folder
process_stocks()
