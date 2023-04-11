from src.Environments.Connect4 import *
from src.Agents.ExpectedSarsa import *
import pprint

env = Connect4Environment()
agent = ExpectedSarsa(verbose = 2)
n_episodes = 5000

agent.train_n_episodes(env, n_episodes)

pprint.pprint(agent.action_state_value_dictionary)
print(agent.action_counts)