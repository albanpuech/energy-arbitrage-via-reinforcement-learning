import gym
from gym import spaces
import numpy as np
import pandas as pd
from tqdm import tqdm
from env import get_dataset as get_dataset


class BatteryDiscrete(gym.Env):
    def __init__(
        self, df, render_mode=None, k=5, NEC=10**5, nbins=None, start_hour=0,
        discrete_cols=None
    ):

        self.NEC = NEC
        self.E1H = NEC / 2
        self.k = k
        self.df = df
        self.start_hour = start_hour

        # discrete prices and its derivatives
        self.discrete_cols = discrete_cols if discrete_cols is not None else [
            "dprice", "dprice_der1", "dprice_der2"]
        self.value_arrays = dict()
        for col in self.discrete_cols:
            self.value_arrays[col] = self.df[col].to_numpy(dtype=int)

        self.scaled_price = self.df.scaled_price.to_numpy()
        self.mean_scaled_price = self.df.scaled_price.rolling(
            self.k).mean().to_numpy()

        self.n_hours = len(self.df)
        self.SOC = np.zeros(self.n_hours, dtype=int)
        self.schedule = np.zeros(self.n_hours)

        # number of bins that the price, first derivative, and second derivative are divided into
        self.nbins = nbins if nbins is not None else [10, 10, 10]
        # additional 3 is for the state of charge which can be between 0, half full, or full (E1H = NEC/2)
        self.observation_space = spaces.MultiDiscrete(list(self.nbins) + [3])

        # We have 3 actions, corresponding to "charge, hold, discharge"
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        obs = []
        for col in self.discrete_cols:
            obs.append(self.value_arrays[col][self.hour])
        obs.append(self.SOC[self.hour])
        return obs

    def _get_info(self):
        df_optim = self.df.copy()
        df_optim["schedule"] = self.schedule
        df_optim["SOC"] = (self.SOC * 100) / 2

        return df_optim

    def reset(self, seed=None, options=None):
        self.hour = self.start_hour
        observation = self._get_obs()
        # info = self._get_info()

        return observation

    def step(self, action):
        if action == 0:  # discharge
            self.SOC[self.hour] = max(0, (self.SOC[self.hour - 1] - 1))

        elif action == 1:  # hold
            self.SOC[self.hour] = self.SOC[self.hour - 1]

        elif action == 2:  # charge
            self.SOC[self.hour] = min(2, (self.SOC[self.hour - 1] + 1))

        self.schedule[self.hour - 1] = (
            (self.SOC[self.hour] - self.SOC[self.hour - 1]) * self.NEC / 2
        )

        reward = (action - 1) * (
            self.mean_scaled_price[self.hour] - self.scaled_price[self.hour]
        )

        self.hour += 1
        terminated = self.hour == self.n_hours
        obs = self._get_obs() if not terminated else None

        return obs, reward, terminated, {}

    def test(self, model=None):

        obs = self.reset()
        reward_list = []
        for i in tqdm(range(len(self.df))):
            action, _states = (
                model.predict(obs, deterministic=True)
                if model
                else (self.action_space.sample(), None)
            )
            obs, reward, done, _ = self.step(action)
            reward_list.append(reward)
            if done:
                break

        return sum(reward_list), self._get_info()
