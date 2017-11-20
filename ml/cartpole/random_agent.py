import logging

import numpy as np

from ml.runner.base import Runner
from ml.agent.base import Agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CartpoleAgent(Agent):
    def __init__(self, weights=None):
        super(CartpoleAgent, self).__init__()

        if weights is None:
            weights = np.random.rand(4) * 2 - 1

        self.weights = weights

    def get_action(self, observation):
        weighted_observation = np.matmul(self.weights, observation)

        if weighted_observation < 0:
            action = 0

        else:
            action = 1

        logger.debug('action %s obs %s', action, weighted_observation)
        return action


def run_experiments():
    results = []
    env_name = 'CartPole-v0'
    num_experiments = 100

    for num_experiment in range(num_experiments):
        num_iterations = 50000
        agent = CartpoleAgent()
        runner = Runner(env_name, num_iterations=num_iterations, agent=agent, max_reward=200)
        reward, num_steps = runner.run()
        results.append({'score': reward, 'num_iterations': num_experiment, 'weights': agent.weights})

        if reward == 200:
            print('Got result in %s experiments', num_experiment)
            break

    return results


def main():
    run_experiments()


if __name__ == '__main__':
    main()
