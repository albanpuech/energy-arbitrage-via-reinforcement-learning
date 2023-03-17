# Battery Scheduling Algorithm for Energy Arbitrage via Reinforcement Learning

## Abstract

As the share of renewable energy sources increases in the electricity market, new solutions are needed to build a flexible and reliable grid that helps at aligning consumption with production. Energy arbitrage with battery storage systems supports renewable energy integration into the grid by shifting demand and increasing the overall utilization of power production systems. In this project, we propose an optimized scheduling algorithm for energy arbitrage via Reinforcement Learning. The algorithm optimally schedules the charge and discharge operations associated with the most profitable trading strategy. We compare our algorithm to a baseline based on a linear programming formulation and report similar performances. We finally illustrate the pitfalls of relying on a reinforcement learning approach.

## Data and training

We use the data available from the European Hourly Day-Ahead Electricity Price [dataset](https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/). This can be downloaded from [here](https://drive.google.com/file/d/1JPYYUoqVU-0NLB9bY6ElcZBqi67KKQ-I/view?usp=share_link).For training and evaluation, we use the pricing data from the years 2020-2022 in Germany. Data from 2021 is used for training, 2022 for testing and 2020 as the evaluation set.

## Project setup instructions

1. Clone the repository.
2. Download the data from the above link and place it in a directory called `data`.
3. Create a Python 3.10 virtual environment and install the required packages using the `requirements.txt` file.
4. You should be able to execute all the notebooks now!

## Project structure

1. `env_continuous.py`: contains an implementation of our battery electricity trading reinforcement learning environment as an Open AI Gym environemnt.
2. `dataset.py`: contains functions to extract data and preprocess it into a dataframe with appropriate column names and indices.
3. `plot.py`: contains functions to plot the graph of a given trading schedule and the daily profits achieved.
4. `historical_hourly_prices_schedule.ipynb`: implements LP-PRED and LP-OPTIM algorithms and evaluates their performance which we use as our baseline models. `baseline_evaluation.ipynb` contains all the graphs evaluating these baseline algorithms.
5. `env_continuous_nb.ipynb`: the main notebook which was used to run experiments trying out different RL algorithms on the `Battery` Gym environment. Experiments were recorded on Comet ML. `model_evaluation.ipynb` contains the graphs evaluating the best performing model.
6. `logs`, `logs2`, and `logs3` folders contain the best performing trained models as `stablebaselines3` models.

## Appendix

### Q-learning

The `q-learning` folder contains a discrete version of the Gym environment upon which we experimented and evaluated the performance of Q-learning algorithms.
