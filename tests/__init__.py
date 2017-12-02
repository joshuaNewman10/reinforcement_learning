import os


def get_data_path(file_name):
    directory = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(directory, 'data', file_name)
