class ActionProvider:
    def __init__(self, action_indices, action_names, action_values):
        self._action_indices = action_indices
        self._action_names = action_names
        self._action_values = action_values

        self._name_index_map = self._create_mapping(action_names, action_indices)
        self._index_name_map = self._create_mapping(action_indices, action_names)
        self._index_value_map = self._create_mapping(action_indices, action_values)

    def _create_mapping(self, a_items, b_items):
        mapping = {}

        for a, b in zip(a_items, b_items):
            mapping[a] = b

        return mapping

    def get_action_value_from_index(self, index):
        return self._index_value_map[index]

    def get_action_name_from_index(self, index):
        return self._index_name_map[index]

    def get_action_index_from_name(self, name):
        return self._name_index_map[name]
