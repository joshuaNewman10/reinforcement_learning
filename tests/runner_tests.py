import unittest

from mock import Mock

from ml.agent.base import Agent
from ml.runner.env.base import Runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = Runner(
            env_name='',
            agent=Mock(spec=Agent),
            max_steps=10
        )
