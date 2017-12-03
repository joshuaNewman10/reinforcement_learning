import logging
import os
import random
from collections import deque

import numpy as np
from keras.layers import Dense, Flatten, Activation, Conv2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras import backend as K

from ml.racer.config import ACTION_NAMES, ACTION_INDICES, ACTION_VALUES
from ml.agent.base import Agent
from ml.runner.experiment.base import ExperimentRunner
from ml.provider.action import ActionProvider
from ml.runner.env.atari import AtariRunner

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class QLearningAgent(Agent):
    def __init__(self, action_provider, num_channels=None, observation_state_size=None, action_size=None, batch_size=64,
                 verbose=True, is_trainable=True):

        super(QLearningAgent, self).__init__(is_trainable=is_trainable)

        self.observation_shape = observation_state_size
        self.action_provider = action_provider
        self.action_size = action_size
        self.num_channels = num_channels

        self._verbose = verbose

        self.batch_size = batch_size
        self.env_step_history = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.failure_penalty = -100
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
            action_index = self.action_provider.get_random_action_index()
            logger.warning('Random action index %s', action_index)
        else:
            action_index = self.predict(current_observation)
            logger.warning('Sending predicted action from agent %s', action_index)

        action = self.action_provider.get_action_value_from_index(action_index)
        logger.warning('Sending action %s', action)
        action_n = [action]
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
        num_samples = 1
        observation_shape = observation.shape

        if observation_shape != self.observation_shape:
            logger.error('Invalid input shape %s', observation_shape)
            raise ValueError()

        height, width = observation_shape
        logger.warning('Formatting obs with dimensions H: %s W: %s', height, width)

        return np.reshape(observation, [num_samples, height, width, self.num_channels])

    def train(self):
        X = []
        y = []
        env_state_mini_batch = random.sample(self.env_step_history, k=self.batch_size)

        for env_state in env_state_mini_batch:
            observation = env_state['observation']
            # num_samples = 1
            # observation_shape = observation.shape
            # height, width = observation_shape
            # channels = 1

            action = env_state['action']
            reward = env_state['reward']
            next_observation = env_state['next_observation']
            done = env_state['done']

            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_observation)[0]))
            future_discounted_reward = self._predict_future_discounted_reward(observation, target, action)
            # self.model.fit(observation, future_discounted_reward, epochs=1, verbose=1)
            X.append(observation)
            y.append(future_discounted_reward)

            logger.warning('Training with future reward %s', future_discounted_reward)

        observation_shape = self.observation_shape
        width, height = observation_shape

        X = np.reshape(X, [len(X), height, width, self.num_channels])
        y = np.reshape(y, [len(y), self.action_size])
        logger.warning('Training with X shape %s y shape ', X.shape, y.shape)

        self.model.fit(X, y, epochs=1, verbose=1)
        self._update_epsilon()

    def _predict_future_discounted_reward(self, observation, target, action):
        target_future_discounted_rewards = self.model.predict(x=observation)
        action_name = self._get_action_name(action)

        action_index = self.action_provider.get_action_index_from_name(action_name)
        logger.warning('Training with future reward from action %s Index %s rewards %s', action_name, action_index,
                       target_future_discounted_rewards)

        # action_space = [0, 0, 0, 0]
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
        num_channels = self.num_channels
        num_actions = self.action_size

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=(height, width, num_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )

        if self._verbose:
            print(model.summary())

        return model

    def _store_env_step(self, action, current_observation, previous_observation, reward, done, info):
        if not self.should_store_env_step(action, current_observation, previous_observation, reward, done, info):
            return

        if done:
            reward = self.failure_penalty

        current_observation = self.format_observation(current_observation)
        previous_observation = self.format_observation(previous_observation)

        logger.warning('Appending action %s reward %s done %s info %s', action, reward, done, info)

        self.env_step_history.append(dict(
            observation=previous_observation,
            action=action,
            reward=reward,
            next_observation=current_observation,
            done=done,
            info=info
        ))

    def load(self, file_path):
        self.model = load_model(file_path)

    def save(self, file_path):
        logger.info('Saving model to file %s', file_path)
        self.model.save(file_path)

    def should_store_env_step(self, action, current_observation, previous_observation, reward, done, info):
        return info['env_has_started'] == True


def main():
    env_name = 'flashgames.NeonRace-v0'
    max_reward = None
    max_steps = 5000
    num_runs = 1000
    screen_height = 764
    screen_width = 1024
    num_channels = 1

    K.set_image_data_format('channels_last')  # TF dimension ordering in this code
    model_file_path = os.environ['MODEL_FILE_PATH']

    action_provider = ActionProvider(
        action_values=ACTION_VALUES,
        action_names=ACTION_NAMES,
        action_indices=ACTION_INDICES
    )

    agent = QLearningAgent(action_provider=action_provider)

    runner = AtariRunner(
        env_name,
        agent=agent,
        max_reward=max_reward,
        max_steps=max_steps,
        reset_on_done=False,
        run_in_docker=True
    )

    observation_shape = (screen_height, screen_width)
    action_size = len(ACTION_VALUES)
    action_space = runner.action_space

    agent.actions = action_space
    agent.action_size = action_size
    agent.observation_shape = observation_shape
    agent.num_channels = num_channels

    model = agent.build_model()
    agent.model = model

    experiment_runner = ExperimentRunner()
    results = experiment_runner.run(
        env_runner_cls=AtariRunner,
        agent=agent,
        env_name=env_name,
        max_reward=max_reward,
        max_steps=max_steps,
        num_runs=num_runs,
        model_file_path=model_file_path
    )

    logger.warning('Results %s', results)


if __name__ == '__main__':
    main()
