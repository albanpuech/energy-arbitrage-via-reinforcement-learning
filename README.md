# Battery Scheduling Algorithm for Energy Arbitrage via Reinforcement Learning

## Abstract

As the share of renewable energy sources increases in the electricity market, new solutions are needed to build a flexible and reliable grid that helps at aligning consumption with production. Energy arbitrage with battery storage systems supports renewable energy integration into the grid by shifting demand and increasing the overall utilization of power production systems. In this project, we propose an optimized scheduling algorithm for energy arbitrage via Reinforcement Learning. The algorithm optimally schedules the charge and discharge operations associated with the most profitable trading strategy. We compare our algorithm to a baseline based on a linear programming formulation and report similar performances. We finally illustrate the pitfalls of relying on a reinforcement learning approach.

## Data and training

We use the data available from the European Hourly Day-Ahead Electricity Price [dataset](https://ember-climate.org/data-catalogue/european-wholesale-electricity-price-data/). For training and evaluation, we use the pricing data from the years 2020-2022 in Germany. Data from 2021 is used for training, 2022 for testing and 2020 as the evaluation set.
