import logging

import numpy as np

from ml.env.base import EnvRunner
from ml.agent.base import Agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CartpoleAgent(Agent):
    def __init__(self, decision_params=None):
        super(CartpoleAgent, self).__init__()

        if decision_params is None:
            decision_params = np.random.rand(4) * 2 - 1

        self.decision_params = decision_params

    def get_action(self, observation):
        weighted_observation = np.matmul(self.decision_params, observation)

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
        runner = EnvRunner(env_name, num_iterations=num_iterations, agent=agent, max_reward=200)
        reward, num_steps = runner.run()
        if reward == 200:
            print('Got result in %s experiments', num_experiment)
            break
        results.append({'score': reward, 'num_steps': num_steps, 'params': agent.decision_params})

    #print(sorted(results, key=lambda x: x['score']))


def main():
    run_experiments()


if __name__ == '__main__':
    main()
