import logging
import random
import numpy as np
import time

from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D
from keras.optimizers import Adam

from ml.agent.base import Agent
from ml.runner.atari import AtariRunner
from ml.provider.action.base import ActionProvider

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class QLearningAgent(Agent):
    def __init__(self, action_provider, observation_state_size=None, action_size=None, batch_size=32, verbose=True):
        super(QLearningAgent, self).__init__()

        self.observation_shape = observation_state_size
        self.action_provider = action_provider
        self.action_size = action_size

        self._verbose = verbose

        self.batch_size = batch_size
        self.env_step_history = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.failure_penalty = - 10
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
            action_index = 0
        else:
            action_index = self.predict(current_observation)

        action = self.action_provider.get_action_value_from_index(action_index)
        action_n = [action]
        logger.warning('Sending action from agent %s', action)
        return action_n

    def has_sufficient_training_data(self):
        return len(self.env_step_history) > self.batch_size

    def predict(self, observation):
        observation = self.format_observation(observation)

        predictions = self.model.predict(x=observation)
        logger.warning("DEBUG PREDICTIONS %s", predictions)
        prediction = predictions[0]
        return np.argmax(prediction)

    def format_observation(self, observation):
        # train_x.shape is (num_sample, height, width, channel) for TensorFlow backend
        # train_x.shape is (num_sample, channels, height, width) for Theano backend
        num_samples = 1
        observation_shape = observation.shape
        height, width = observation_shape
        channels = 1

        return np.reshape(observation, [num_samples, height, width, channels])

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
            logger.warning('Training with future rewrd %s', future_discounted_reward)
            self.model.fit(observation, future_discounted_reward, epochs=1, verbose=1)
        self._update_epsilon()

    def _predict_future_discounted_reward(self, observation, target, action):
        target_future_discounted_rewards = self.model.predict(x=observation)
        action_name = self._get_action_name(action)
        action_index = self.action_provider.get_action_index_from_name(action_name)
        logger.warning('JJDEBUG ACTION %s Index %s rewards %s', action_name, action_index, target_future_discounted_rewards)
        #action_space = [0, 0, 0, 0]
        target_future_discounted_rewards[0][action_index] = target
        return target_future_discounted_rewards

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _get_action_name(self, action):
        set_action = None

        for part in action[0]:
            part_type, name, set_value = part
            if set_value == True:
                set_action = name

        return set_action

    def build_model(self):
        observation_shape = self.observation_shape
        height, width = observation_shape
        channels = 1

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=(height, width, channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )

        if self._verbose:
            print(model.summary())

        return model

    def _store_env_step(self, action, current_observation, previous_observation, reward, done, info):
        if action is None or reward == 0:
            return

        if done:
            reward = self.failure_penalty

        current_observation = self.format_observation(current_observation)
        previous_observation = self.format_observation(previous_observation)

        logger.warning('Appending action %s reward %s done %s info %s', action, reward, done, info)

        self.env_step_history.append(
            {
                'observation': previous_observation,
                'action': action,
                'reward': reward,
                'next_observation': current_observation,
                'done': done,
                'info': info
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


def get_actions():
    action_names = ['ArrowUp', 'ArrowLeft', 'ArrowRight']
    action_values = [
        [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)],
        [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)],
        [('KeyEvent', 'ArrowUp', False), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]
    ]

    action_indices = [0, 1, 2]
    return action_names, action_values, action_indices


def main():
    env_name = 'flashgames.NeonRace-v0'
    max_reward = 5000
    max_steps = 10000000000
    num_episodes = 1000

    episode_data = []

    action_names, action_values, action_indices = get_actions()
    action_provider = ActionProvider(
        action_values=action_values,
        action_names=action_names,
        action_indices=action_indices
    )
    agent = QLearningAgent(action_provider=action_provider)
    runner = AtariRunner(env_name, agent=agent, max_reward=max_reward, max_steps=max_steps, reset_on_done=False,
                         run_in_docker=True)
    observation_shape = runner.observation_shape
    action_size = len(action_values)
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
