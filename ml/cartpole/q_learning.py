import logging
import random
import numpy as np
import time

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from ml.agent.base import Agent
from ml.runner.base import Runner

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class QLearningAgent(Agent):
    def __init__(self, actions=None, observation_state_shape=None, action_size=None, batch_size=32, verbose=True):
        super(QLearningAgent, self).__init__()

        self.actions = actions
        self.observation_shape = observation_state_shape
        self.action_size = action_size

        self._verbose = verbose

        self.batch_size = batch_size
        self.env_step_history = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = None

    def get_action(self, action, current_observation, previous_observation, reward, done, info):
        self._store_env_step(
            action=action,
            done=done,
            previous_observation=previous_observation,
            current_observation=current_observation,
            reward=reward,
            info=info
        )

        random_decimal = np.random.rand()

        if random_decimal <= self.epsilon:
            action = self.actions.sample()
        else:
            action = self.predict(current_observation)

        return action

    def has_sufficient_training_data(self):
        return len(self.env_step_history) > self.batch_size

    def predict(self, observation):
        observation = self.format_observation(observation)

        # [[0.4, 0.6]
        predictions = self.model.predict(x=observation)
        prediction = predictions[0]
        return np.argmax(prediction)

    def format_observation(self, observation):
        return np.reshape(observation, [1, self.observation_shape])

    def train(self):
        env_state_mini_batch = random.sample(self.env_step_history, k=self.batch_size)

        for env_state in env_state_mini_batch:
            observation = env_state['observation']
            action = env_state['action']
            reward = env_state['reward']
            next_observation = env_state['next_observation']
            done = env_state['done']

            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_observation)[0]))

            future_discounted_reward = self._predict_future_discounted_reward(observation, target, action)
            self.model.fit(observation, future_discounted_reward, epochs=1, verbose=0)
        self._update_epsilon()

    def _predict_future_discounted_reward(self, observation, target, action):
        target_future_discounted_rewards = self.model.predict(x=observation)
        target_future_discounted_rewards[0][action] = target
        return target_future_discounted_rewards

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )

        if self._verbose:
            print(model.summary())

        return model

    def _store_env_step(self, action, current_observation, previous_observation, reward, done, info):
        if action is None:
            return

        if done:
            reward = -10

        current_observation = self.format_observation(current_observation)
        previous_observation = self.format_observation(previous_observation)

        self.env_step_history.append(
            {
                'observation': previous_observation,
                'action': action,
                'reward': reward,
                'next_observation': current_observation,
                'done': done
            }
        )

    def load(self, file_path):
        self.model.load_weights(file_path)

    def save(self, file_path):
        self.model.save_weights(file_path)


def run_episode(runner, episode_num, verbose=False):
    if verbose:
        logger.warning('Running episode num %f', episode_num)

    start = time.time()
    reward, num_steps = runner.run()
    end = time.time()
    run_time = end - start

    if verbose:
        logger.warning('Ran episode in %s seconds: %s reward %s num_steps', run_time, reward, num_steps)

    return reward, num_steps


def compute_avg_reward(episode_data):
    rewards = [episode['reward'] for episode in episode_data]
    return np.mean(rewards)


def compute_avg_num_steps(episode_data):
    num_steps = [step_count['num_steps'] for step_count in episode_data]
    return np.mean(num_steps)


def main():
    env_name = 'CartPole-v1'
    max_reward = 1000
    max_steps = 1000
    num_episodes = 1000

    episode_data = []

    agent = QLearningAgent()
    runner = Runner(env_name, agent=agent, max_reward=max_reward, max_steps=max_steps, reset_on_done=False)
    observation_shape = runner.observation_shape #tuple of shape (4,)
    observation_shape = observation_shape[0]
    action_size = runner.action_size
    action_space = runner.action_space

    agent.actions = action_space  # see if can get dynamically
    agent.action_size = action_size
    agent.observation_shape = observation_shape

    model = agent.build_model()
    agent.model = model

    for episode_num in range(num_episodes):
        reward, num_steps = run_episode(runner, episode_num)
        episode_data.append(dict(reward=reward, num_steps=num_steps))
        logger.warning('Episode %s Score %s Steps %s epis %s', episode_num, reward, num_steps, agent.epsilon)
        if agent.has_sufficient_training_data():
            agent.train()

    avg_reward = compute_avg_reward(episode_data)
    avg_num_steps = compute_avg_num_steps(episode_data)
    logger.warning('Avg Reward %s Avg Num Steps %s', avg_reward, avg_num_steps)


if __name__ == '__main__':
    main()
