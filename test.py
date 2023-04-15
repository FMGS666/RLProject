from src.Environments.Connect4.Connect4 import *
from src.Environments.TicTacToe.TicTacToe import *
from src.Agents.ExpectedSarsa import *
import pprint

env = Connect4Environment()
agent = ExpectedSarsa(verbose = 1)
n_episodes = 500


agent.train_n_episodes(env, n_episodes, dump = False)