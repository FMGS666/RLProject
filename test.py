from src.Environments.Connect4 import *
from src.Agents.ExpectedSarsa import *
import pprint

env = Connect4Environment()
agent = ExpectedSarsa(verbose = 2)
n_episodes = 5

agent.train_n_episodes(env, n_episodes, dump = False)

action_state_value_dictionary = {key: list([float(val) for val in value]) for key, value in agent.action_state_value_dictionary.items()}

print(type(list(action_state_value_dictionary.values())[0][0]))
