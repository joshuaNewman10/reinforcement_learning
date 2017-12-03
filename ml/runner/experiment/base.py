import os
import numpy as np
import logging
import time

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, verbose=True):
        self.experiment_data = []

        self.session_num = 0
        self._verbose = verbose

    def _run_session(self, env_runner):
        if self._verbose:
            logger.warning('Running episode num %f', self.session_num)

        start = time.time()
        reward, num_steps = env_runner.run()
        end = time.time()

        run_time = end - start

        if self._verbose:
            logger.warning('Ran episode in %s seconds: %s reward %s num_steps', run_time, reward, num_steps)

        self.experiment_data.append(dict(reward=reward, session_num=self.session_num))
        return reward, self.session_num

    def run(self, env_runner_cls, agent, env_name, max_reward, max_steps, num_runs, model_file_path=None, **kwargs):
        for _ in range(num_runs):
            env_runner = env_runner_cls(
                env_name,
                agent=agent,
                max_reward=max_reward,
                max_steps=max_steps,
                run_in_docker=True,
                **kwargs
            )

            reward, num_steps = self._run_session(env_runner)
            self.experiment_data.append(dict(reward=reward, session_num=self.session_num, num_steps=num_steps))

            logger.warning('Episode %s Score %s Steps %s Session %s', self.session_num, reward, num_steps,
                           agent.epsilon)

            if agent.is_trainable and agent.has_sufficient_training_data():
                agent.train()

        if model_file_path:
            agent.save_model(model_file_path)

        return self._compute_experiment_data_stats()

    def _compute_experiment_data_stats(self):
        avg_reward = self._compute_avg_reward()
        avg_num_steps = self._compute_avg_num_steps()

        return dict(
            experiment_data=self.experiment_data,
            num_sessions=self.session_num,
            avg_reward=avg_reward,
            avg_num_steps=avg_num_steps
        )

    def _compute_avg_reward(self):
        rewards = [episode['reward'] for episode in self.experiment_data]
        return np.mean(rewards)

    def _compute_avg_num_steps(self):
        num_steps = [step_count['num_steps'] for step_count in self.experiment_data]
        return np.mean(num_steps)
