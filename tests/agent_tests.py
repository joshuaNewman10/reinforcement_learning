import unittest

from mock import Mock
from ml.agent.base import Agent



class AgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()

    def test_update_reward(self):
        self.agent.update_reward(reward=20, done=False)
        self.assertEqual(self.agent.total_reward, 20)

        self.agent.update_reward(reward=20, done=False)
        self.assertEqual(self.agent.total_reward, 40)

    def test_rest_reward(self):
        self.agent.update_reward(reward=20, done=False)
        self.agent.reset_reward()
        self.assertEqual(self.agent.total_reward, 0)

    def test_reset(self):
        self.agent.reset_reward = Mock()
        self.agent.update_reward(reward=20, done=False)
        self.agent.reset()

        self.agent.reset_reward.assert_called_once()



