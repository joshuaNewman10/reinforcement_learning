import numpy as np

from ml.runner.base import Runner
from ml.util.image import convert_image_to_greyscale, downsample_image



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
        if observation is None or not any(observation):
            screen_image = self.black_screen
            return screen_image
        else:
            screen_image = observation['vision']
            image = screen_image
            screen_image = downsample_image(image, scale=self.downsample_rate)
            screen_image = convert_image_to_greyscale(screen_image)
            return screen_image

    def run(self):
        action = None
        info = None
        previous_observation = None

        done = False
        reward = 0
        step_num = 0

        observation_n = self.env.reset()
        observation = observation_n[0]
        observation = self.preprocess_observation(observation)
        self.agent.reset_reward()

        while True:
            self.env.render()
            step_num += 1

            action = self.agent.get_action(
                action=action,
                previous_observation=previous_observation,
                current_observation=observation,
                reward=reward,
                done=done,
                info=info
            )

            previous_observation = observation
            observation_n, reward_n, done_n, info = self.env.step(action)
            observation = observation[0]
            reward = reward_n[0]
            done = done_n[0]

            observation = self.preprocess_observation(observation)
            self.agent.update_reward(reward, done)

            if self.max_reward_reached():
                break

            if done and self.reset_on_done:
                observation = self.env.reset()
                self.agent.reset_reward()
                step_num = 0

            elif done and not self.reset_on_done:
                break

        self.agent.get_action(
            action=action,
            previous_observation=previous_observation,
            current_observation=observation,
            reward=reward,
            done=done,
            info=info
        )

        return self.agent.total_reward, step_num


    def _downsample_observation_shape(self, observation_state_shape):
        downsampled_shape = [int(dimension * self.downsample_rate) for dimension in observation_state_shape]
        return downsampled_shape
