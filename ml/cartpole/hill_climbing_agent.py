import logging
import os

import numpy as np

from ml.agent.base import Agent
from ml.runner.env.base import Runner

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class HillClimbingAgent(Agent):
    def __init__(self, weights=None, noise_scaling=None):
        super(HillClimbingAgent, self).__init__()

        if weights is None:
            weights = np.random.rand(4) * 2 - 1

        if noise_scaling is None:
            noise_scaling = 0.1

        self.noise_scaling = noise_scaling
        self.weights = weights

    def get_action(self, action, current_observation, previous_observation, reward, done, info):
        self.weights = self.weights + (np.random.rand(4) * 2 - 1) * self.noise_scaling
        weighted_observation = np.matmul(self.weights, current_observation)

        if weighted_observation < 0:
            action = 0

        else:
            action = 1

        logger.debug('action %s obs %s', action, weighted_observation)
        return action


def run_experiments():
    best_reward = 0
    best_weights = np.random.rand(4) * 2 - 1
    env_name = 'CartPole-v0'
    monitor_dir = os.getenv('GYM_MONITOR_DIR')
    num_experiments = 1000
    num_experiment = 0

    for num_experiment in range(num_experiments):
        num_iterations = 200
        agent = HillClimbingAgent(weights=best_weights)
        runner = Runner(env_name, max_steps=num_iterations, agent=agent, max_reward=200, monitor_dir=monitor_dir)
        reward, num_steps = runner.run()
        weights = agent.weights

        if reward > best_reward:
            best_reward = reward
            best_weights = weights

        agent.weights = best_weights

        if reward >= 200:
            break

    logger.warning('Best weights %s best reward %s num experiment %s', best_weights, best_reward, num_experiment)
    return best_weights, best_reward


def main():
    run_experiments()


if __name__ == '__main__':
    main()
