import gym
from gym import spaces
import numpy as np
import pandas as pd
from tqdm import tqdm


class Battery(gym.Env):
    def __init__(
        self, df, cols, render_mode=None, NEC=10**5, start_hour=0, reward_function=None
    ):

        self.NEC = NEC
        self.E1H = NEC / 2
        self.df = df
        self.start_hour = start_hour

        # discrete prices and its derivatives
        self.cols = cols
        self.value_arrays = dict()
        for col in self.cols:
            self.value_arrays[col] = self.df[col].to_numpy()

        self.buying_price = 0

        self.n_hours = len(self.df)
        self.SOC = np.zeros(self.n_hours)
        self.schedule = np.zeros(self.n_hours)
        self.cash_in_hand = np.zeros(self.n_hours)

        # additional 3 is for the state of charge which can be between 0, half full, or full (E1H = NEC/2)
        self.observation_space = spaces.Box(low=np.array([-np.inf for _ in range(len(
            cols) + 1)]), high=np.array([np.inf for _ in range(len(cols) + 1)]), shape=(len(cols)+1,), dtype=np.float64)

        self.reward_function = reward_function

        # We have 3 actions, corresponding to "charge, hold, discharge"
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        obs = []
        for col in self.cols:
            obs.append(self.value_arrays[col][self.hour])

        obs.append(self.SOC[self.hour])
        return np.array(obs)

    def _get_info(self):
        df_optim = self.df.copy()
        df_optim["schedule"] = self.schedule
        df_optim["SOC"] = (self.SOC * 100) / 2

        return df_optim

    def reset(self, seed=None, options=None):
        self.hour = self.start_hour
        self.SOC = np.zeros(self.n_hours)
        self.schedule = np.zeros(self.n_hours)

        self.buying_price = 0
        self.cash_in_hand = np.zeros(self.n_hours)
        observation = self._get_obs()
        # info = self._get_info()

        return observation

    def _get_valuation(self, hour):
        return self.cash_in_hand[hour] + (self.SOC[hour] * self.df.price[hour])

    def _get_reward(self, action):
        if self.reward_function is None:
            return self._get_valuation(self.hour) - self._get_valuation(self.hour - 1)
        return self.reward_function(self, action)

    def step(self, action):
        if action == 0:  # discharge
            self.SOC[self.hour] = max(0, (self.SOC[self.hour - 1] - 1))
            if self.SOC[self.hour] == 0:
                self.buying_price = 0

        elif action == 1:  # hold
            self.SOC[self.hour] = self.SOC[self.hour - 1]

        elif action == 2:  # charge
            self.SOC[self.hour] = min(2, (self.SOC[self.hour - 1] + 1))
            if self.SOC[self.hour - 1] == 0:
                self.buying_price = self.df.price[self.hour]
            elif self.SOC[self.hour - 1] == 1:
                self.buying_price = (self.buying_price +
                                     self.df.price[self.hour])/2.0
            else:
                self.buying_price = self.buying_price

        self.schedule[self.hour - 1] = (
            (self.SOC[self.hour] - self.SOC[self.hour - 1]) * self.NEC / 2
        )
        self.cash_in_hand[self.hour] = self.cash_in_hand[self.hour - 1] - (
            (self.SOC[self.hour] - self.SOC[self.hour - 1])*self.df.price[self.hour-1]) - \
            abs(self.SOC[self.hour] - self.SOC[self.hour - 1]) * \
            self.df.vgc[self.hour-1]

        reward = self._get_reward(action)

        terminated = self.hour == self.n_hours - 1
        obs = self._get_obs() if not terminated else None
        self.hour += 1

        return obs, reward, terminated, {}

    def test(self, model=None):

        obs = self.reset()
        reward_list = []
        for i in range(len(self.df)):
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
