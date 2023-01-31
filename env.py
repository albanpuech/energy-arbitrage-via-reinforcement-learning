import gym
from gym import spaces
import numpy as np
import pandas as pd
from tqdm import tqdm



def get_dataset(path="data/european_wholesale_electricity_price_data_hourly.csv", year="2016",country="Germany", usecols=["Datetime (Local)", "Price (EUR/MWhe)", "Country"]):
    df = pd.read_csv(path,usecols=usecols)
    df = df[df.Country == country]
    df = df[df["Datetime (Local)"] > "2020-01-01 00:00:00"]
    df.drop(["Country"], axis=1, inplace=True)
    df.rename({"Datetime (Local)": "timestamp",
                "Price (EUR/MWhe)": "price"}, axis=1,inplace=True,errors="raise")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S")
    df.reset_index(drop=True, inplace=True)
    return df


class Battery(gym.Env):

    def __init__(self,  df,render_mode=None,k=5, NEC=10**5):

        self.NEC = NEC
        self.E1H = NEC/2
        self.k = k
        self.df = df 
        self.scaled_price = self.df.scaled_price.to_numpy()
        self.mean_scaled_price = self.df.scaled_price.rolling(self.k).mean().to_numpy()
        self.n_hours = len(self.df)
        self.SOC = np.zeros(self.n_hours)
        self.schedule = np.zeros(self.n_hours)

        self.observation_space = spaces.Box(low=np.array([-1000 for _ in range(k)] + [0.0 for _ in range(k)]), high=np.array([1000 for _ in range(k)] + [1.0 for _ in range(k)]), shape=(k * 2,),dtype=np.float64)

        # We have 3 actions, corresponding to "charge, hold, discharge"
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        return np.concatenate([self.scaled_price[self.hour-self.k+1:self.hour+1],
                          self.SOC[self.hour-self.k:self.hour]])

    def _get_info(self):
        df_optim = self.df.copy()
        df_optim["schedule"] = self.schedule
        df_optim["SOC"] = self.SOC * 100
        
        return df_optim
    

    def reset(self, seed=None, options=None):
        self.hour = self.k
        observation = self._get_obs()
        # info = self._get_info()

        return observation

    def step(self, action):
        if action == 0:
            self.SOC[self.hour] = max(
                0,  (self.SOC[self.hour-1]*self.NEC - self.E1H) / self.NEC)

        elif action == 1:
            self.SOC[self.hour] = self.SOC[self.hour-1]

        elif action == 2:
            self.SOC[self.hour] = min(
                1,  (self.SOC[self.hour-1]*self.NEC + self.E1H) / self.NEC)
            
        self.schedule[self.hour] = (self.SOC[self.hour]-self.SOC[self.hour-1]) * self.NEC

        reward = (action-1) * \
            (self.mean_scaled_price[self.hour] - self.scaled_price[self.hour])

        self.hour += 1
        terminated = self.hour == self.n_hours
        obs = self._get_obs() if not terminated else None

        return obs, reward, (self.hour == self.n_hours), {}

    def test(self,model=None) :
        
        obs = self.reset()
        reward_list = []
        for i in tqdm(range(len(self.df))) :
            action, _states = model.predict(obs, deterministic=True) if model else (self.action_space.sample(),None)
            obs, reward, done, _ = self.step(action)
            reward_list.append(reward)
            if done : break
        
        return sum(reward_list), self._get_info()


