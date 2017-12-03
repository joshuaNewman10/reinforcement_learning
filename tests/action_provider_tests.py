import unittest

from mock import Mock
from ml.agent.base import Agent
from ml.provider.action import ActionProvider


class ActionProviderTests(unittest.TestCase):
    def setUp(self):
        self.indices = [0, 1, 2]
        self.values = ['A', 'B', 'C']
        self.names = ['Left', 'Center', 'Right']

        self.provider = ActionProvider(
            action_indices=self.indices,
            action_values=self.values,
            action_names=self.names
        )

    def test_create_mapping(self):
        mapping = self.provider._create_mapping(self.indices, self.values)
        self.assertIsInstance(mapping, dict)
        self.assertEqual(mapping, {0: 'A', 1: 'B', 2: 'C'})

    def test_get_random_action_index(self):
        ix = self.provider.get_random_action_index()
        self.assertIn(ix, self.indices)
