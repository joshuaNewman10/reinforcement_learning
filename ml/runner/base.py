import logging

import gym
import universe

logging.basicConfig(level=logging.DEBUG)


class Runner:
    def __init__(self, env_name, max_steps, agent, max_reward=None, reset_on_done=False, monitor_dir=None):
        self.max_steps = max_steps
        self.max_reward = max_reward
        self.reset_on_done = reset_on_done

        self.agent = agent
        self.env = gym.make(env_name)

        self.action_space = self.get_action_space()
        self.observation_shape = self.observation_state_shape()
        self.action_size = self.get_action_size()

        if monitor_dir:
            gym.wrappers.Monitor(self.env, monitor_dir, force=True)

    def get_action_space(self):
        return self.env.action_space

    def observation_state_shape(self):
        return self.env.observation_space.shape

    def get_action_size(self):
        return self.env.action_space.n

    def max_reward_reached(self):
        if (self.max_reward is not None) and (self.agent.total_reward >= self.max_reward):
            return True

        return False

    def preprocess_observation(self, observation):
        return observation

    def run(self):
        action = None
        info = None
        previous_observation = None

        done = False
        reward = 0
        step_num = 0

        observation = self.env.reset()
        observation = self.preprocess_observation(observation)
        self.agent.reset_reward()

        for step in range(self.max_steps):
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
            observation, reward, done, info = self.env.step(action)
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
