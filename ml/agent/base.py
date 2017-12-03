from abc import abstractmethod


class Agent:
    def __init__(self, is_trainable=False):
        self.total_reward = 0
        self.is_trainable = is_trainable

    @abstractmethod
    def get_action(self, action, current_observation, previous_observation, reward, done, info):
        raise NotImplementedError

    def update_reward(self, reward, done):
        self.total_reward += reward

    def reset(self):
        self.reset_reward()

    def reset_reward(self):
        self.total_reward = 0
