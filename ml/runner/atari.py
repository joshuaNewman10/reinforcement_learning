import os
import logging
import numpy as np

from ml.runner.base import Runner
from ml.util.image import convert_image_to_greyscale, downsample_image

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logger = logging.getLogger(__name__)


class AtariRunner(Runner):
    downsample_rate = 0.2
    def __init__(self, env_name, max_steps, agent, max_reward=None, run_in_docker=False, reset_on_done=False,
                 monitor_dir=None):

        super(AtariRunner, self).__init__(env_name, max_steps, agent, max_reward, reset_on_done, monitor_dir)
        self.black_screen = np.zeros(shape=(self.observation_shape))

        if run_in_docker:
            self.env.configure(remotes=1)

    def get_action_space(self):
        return self.env.action_space

    def observation_state_shape(self):
        observation_state_shape = self.env.env.action_space.screen_shape
        observation_state_shape = self._downsample_observation_shape(observation_state_shape)
        return observation_state_shape

    def get_action_size(self):
        return len(self.env.action_space.keys)

    def preprocess_observation(self, observation):
        observation = observation[0]

        if observation is None or not any(observation):
            screen_image = self.black_screen
            return screen_image
        else:
            screen_image = observation['vision']
            image = screen_image
            #screen_image = downsample_image(image, scale=self.downsample_rate)
            screen_image = convert_image_to_greyscale(screen_image)
            return screen_image

    def preprocess_action(self, action):
        return action[0]

    def preprocess_reward(self, reward):
        return reward[0]

    def preprocess_done(self, done, info):
        return done[0]

    def _downsample_observation_shape(self, observation_state_shape):
        downsampled_shape = [int(dimension * self.downsample_rate) for dimension in observation_state_shape]
        return downsampled_shape
