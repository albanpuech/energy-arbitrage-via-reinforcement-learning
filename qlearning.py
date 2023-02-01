import numpy as np
import gym


class QLearning:
    def __init__(
        self, env: gym.Env, state_shape, nactions, alpha=0.1, gamma=0.9, epsilon=0.1
    ):
        self.env: gym.Env = env
        self.state_shape = state_shape
        self.nactions = nactions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # initialize Q to zeros
        self.Q = np.zeros(shape=tuple(list(state_shape) + [nactions]))

    def learn(self, total_timesteps):
        state = self.env.reset()

        for _ in range(total_timesteps):
            if np.random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.Q[self._get_index(state)])

            next_state, reward, terminated, _ = self.env.step(action)

            if terminated:
                # reset and start again
                state = self.env.reset()
                continue

            old_value = self.Q[self._get_index(state, action)]
            next_max = np.max(self.Q[self._get_index(next_state)])
            new_value = (1 - self.alpha) * old_value + self.alpha * (
                reward + self.gamma * next_max
            )
            # update Q matrix
            self.Q[self._get_index(state, action)] = new_value

            state = next_state

    def predict(self, state, deterministic=False):
        return np.argmax(self.Q[self._get_index(state)]), None

    def _get_index(self, state, action=None):
        if action is None:
            return tuple(state)
        return tuple(state) + (action,)
