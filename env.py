import gym
from gym import spaces
import numpy as np
import pandas as pd


def get_dataset(path="data/european_wholesale_electricity_price_data_hourly.csv", year="2016",country="Germany", usecols=["Datetime (Local)", "Price (EUR/MWhe)", "Country"]):
    df = pd.read_csv(path,usecols=usecols)
    df = df[df.Country == country]
    df = df[df["Datetime (Local)"] > "2020-01-01 00:00:00"]
    df.drop(["Country"], axis=1, inplace=True)
    df.rename({"Datetime (Local)": "timestamp",
                "Price (EUR/MWhe)": "price"}, axis=1,inplace=True,errors="raise")
    df.price = df.price / 10 ** 6
    df = df.reset_index()
    return df


class Battery(gym.Env):

    def __init__(self, render_mode=None, df=None, k=5, NEC=10**5):

        self.NEC = NEC
        self.E1H = NEC/2
        self.k = k
        self.df = df if len(df)>0 else get_dataset()
        self.df["mean_price"] = self.df.price.rolling(self.k).mean()
        self.n_hours = len(self.df)
        self.SOC = np.zeros(self.n_hours)

        self.observation_space = spaces.Box(low=np.array([[0.0, -1] for i in range(
            k)]), high=np.array([[1.0, 2] for i in range(k)]), shape=(k, 2), dtype=np.float32)

        # We have 3 actions, corresponding to "charge, hold, discharge"
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        return np.hstack([self.df.loc[self.hour-self.k+1:self.hour+1, "price"],
                          self.SOC[self.hour-self.k:self.hour]])

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        self.hour = self.k
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if action == 0:
            self.SOC[self.hour] = max(
                0,  (self.SOC[self.hour-1]*self.NEC - self.E1H) / self.NEC)

        elif action == 1:
            self.SOC[self.hour] = self.SOC[self.hour-1]

        elif action == 2:
            self.SOC[self.hour] = min(
                1,  (self.SOC[self.hour-1]*self.NEC + self.E1H) / self.NEC)

        reward = (action-1) * \
            (self.df.mean_price.loc[self.hour] - self.df.price.loc[self.hour])

        self.hour += 1

        return self._get_obs(), reward, (self.hour == self.n_hours)



