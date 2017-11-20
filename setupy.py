from setuptools import setup, find_packages

setup(name='reinforcement',
      version='0.0.1',
      zip_safe=False,
      packages=find_packages(exclude=('deploy', 'tests')),
      )
