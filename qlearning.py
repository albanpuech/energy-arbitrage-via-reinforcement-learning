import numpy as np
import gym
import random


class QLearning:
    def __init__(
        self, env: gym.Env, discrete_cols,price_quantiles, nactions, alpha=0.1, gamma=0.9, epsilon=0.1
    ):
        self.env: gym.Env = env
        self.state_shape = tuple([n_bin for (_,n_bin) in discrete_cols]) + (nactions,)
        self.nactions = nactions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # initialize Q to zeros
        self.Q = np.zeros(shape=tuple(list(self.state_shape) + [nactions]))
        self.Q_update_count = np.zeros(shape=tuple(list(self.state_shape) + [nactions]))

    def learn(self, total_timesteps=None):
        state = self.env.reset()
        t = 0
        while not total_timesteps or t < total_timesteps :
            if np.random.random() < self.epsilon:
                if state[-1] == 0:
                    action = np.random.choice([1, 2],p=[0.1,0.9])
                elif state[-1] == 2:
                    action = np.random.choice([1, 0],p=[0.1,0.9])
                else : action = self.env.action_space.sample()
            else:
                action = np.random.choice(np.flatnonzero(self.Q[self._get_index(
                    state)] == np.max(self.Q[self._get_index(state)])))

            next_state, reward, terminated, _ = self.env.step(action)
            t+=1
            if terminated:
                # reset and start again
                state = self.env.reset()
                if total_timesteps : 
                    continue
                else : return

            old_value = self.Q[self._get_index(state, action)]
            next_max = np.max(self.Q[self._get_index(next_state)])
            new_value = (1 - self.alpha) * old_value + self.alpha * (
                reward + self.gamma * next_max
            )
            # update Q matrix
            self.Q_update_count[self._get_index(state, action)] +=1
            self.Q[self._get_index(state, action)] = new_value

            state = next_state



    
    def learn_SARSA(self, total_timesteps):
        state = self.env.reset()
        action = random.choice([1, 2])
        
        for _ in range(total_timesteps):
            

            next_state, reward, terminated, _ = self.env.step(action)


            if terminated:
                # reset and start again
                next_state = self.env.reset()
                continue

            if np.random.random() < self.epsilon:
                if state[-1] == 0:
                    next_action = random.choice([1, 2])
                elif state[-1] == 2:
                    next_action = random.choice([0, 1])
                else : next_action = self.env.action_space.sample()
            else:
                next_action = np.random.choice(np.flatnonzero(self.Q[self._get_index(next_state)] == np.max(self.Q[self._get_index(next_state)])))




            old_value = self.Q[self._get_index(state, action)]
            new_value = (1 - self.alpha) * old_value + self.alpha * (
                reward + self.gamma * self.Q[self._get_index(next_state,next_action)]
            )
            # update Q matrix
            self.Q_update_count[self._get_index(state, action)] +=1
            self.Q[self._get_index(state, action)] = new_value

            state = next_state
            action = next_action



    def predict(self, state, deterministic=False):
        action, states_ = np.random.choice(np.flatnonzero(self.Q[self._get_index(
            state)] == np.max(self.Q[self._get_index(state)]))), None
        return action, states_

    def _get_index(self, state, action=None):
        if action is None:
            return tuple(state)
        return tuple(state) + (action,)
