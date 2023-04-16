from src.Environments.TicTacToe.TicTacToe import *
from src.Agents.ExpectedSarsa import ExpectedSarsa


n_episodes = 1
env = TicTacToeEnvironment()
agent = ExpectedSarsa((3, 3), debug = 2)

agent.train_n_episodes(env, n_episodes, dump = False)