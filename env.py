import gym
from gym import spaces
import pygame
import numpy as np
import pandas as pd 





class GridWorldEnv(gym.Env):

    def __init__(self, render_mode=None, k=5, NEC=10**5):
        
        self.NEC = NEC
        self.E1H = NEC/2
        self.df = pd.read_csv("dataset.csv")
        self.n_hours = len(self.df)
        self.k = k
        self.SOC = np.zeros(self.n_hours)

        self.observation_space = spaces.Box(low=np.array([[0.0, -1] for i in range(k)]), high=np.array([[1.0, 2] for i in range(k)]), shape=(k,2),dtype=np.float32)
    
        # We have 3 actions, corresponding to "charge, hold, discharge"
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
      return np.hstack([df.loc[self.hour-self.k+1:self.hour+1,"price"],
                               self.SOC[self.hour-self.k:self.hour]])
    def _get_info(self):
      return self.df[self.hour]

    def reset(self, seed=None, options=None):
      self.hour = self.k
      observation = self._get_obs()
      info = self._get_info()
      
      return observation, info

    def step(self,action) :
      if action == 0 :
        self.SOC[self.hour] = max(0,  (self.SOC[self.hour-1]*self.NEC - self.E1H) / self.NEC)
      

      elif action == 1 :
        self.SOC[self.hour] = self.SOC[self.hour-1]
        
      elif action == 2 :
        self.SOC[self.hour] = min(1,  (self.SOC[self.hour-1]*self.NEC + self.E1H) / self.NEC)

      reward = (self.action-1) * (self.df.mean_price[self.hour] - self.df.price[self.hour])

      self.hour +=1


      return self._get_obs(), reward, (self.hour == self.n_hours)

      
      

        
      


    
