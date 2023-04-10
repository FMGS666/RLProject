from src.Environments.Connect4 import *
from src.Agents.ExpectedSarsa import *

env = Connect4Environment()
agent = ExpectedSarsa()
n_episodes = 100

agent.train_n_episodes(env, n_episodes)