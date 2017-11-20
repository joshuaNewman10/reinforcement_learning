import logging

import gym

logging.basicConfig(level=logging.DEBUG)


class Runner:
    def __init__(self, env_name, num_iterations, agent, max_reward=None, reset_on_done=False):
        self.agent = agent
        self.env = gym.make(env_name)
        self.num_iterations = num_iterations
        self.total_reward = 0
        self.max_reward = max_reward
        self.reset_on_done = reset_on_done

    def update_reward(self, reward):
        self.total_reward += reward

    def max_reward_reached(self):
        if (self.max_reward is not None) and (self.total_reward >= self.max_reward):
            return True

        return False

    def run(self):
        observation = self.env.reset()
        iteration_num = 0

        for iteration_num in range(self.num_iterations):
            action = self.agent.get_action(observation)
            observation, reward, done, info = self.env.step(action)
            self.agent.update_reward(reward)
            self.update_reward(reward)

            if self.max_reward_reached():
                break

            if done and self.reset_on_done:
                self.env.reset()

            elif done and not self.reset_on_done:
                break

        return self.total_reward, iteration_num
