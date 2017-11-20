from abc import abstractmethod


class Agent:
    def __init__(self):
        self.total_reward = 0

    @abstractmethod
    def get_action(self, observation):
        raise NotImplementedError

    def update_reward(self, reward):
        self.total_reward += reward
