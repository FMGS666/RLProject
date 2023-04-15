from src.Environments.Connect4.Connect4 import *
from src.Environments.TicTacToe.TicTacToe import *
from src.Agents.ExpectedSarsa import *
import pprint

env = TicTacToeEnvironment()
# print(env.legal_moves)

# env.update_legal_moves((0,0))
# print(env.legal_moves)

agent = ExpectedSarsa((3, 3), debug = 2)
n_episodes = 1

agent.train_n_episodes(env, n_episodes, dump = False)
print(agent.action_state_value_dictionary)
print(env.winner)
print(env.legal_moves)