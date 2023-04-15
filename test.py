from src.Environments.Connect4.Connect4 import *
from src.Environments.TicTacToe.TicTacToe import *
from src.Agents.ExpectedSarsa import *
import pprint

env = Connect4Environment()

agent = ExpectedSarsa()

agent.load('TrainedAgents/ExpectedSarsaConnect4/ExpectedSarsa_eps0.1_gamma0.5_alpha0.1.json')
agent.play_against_random(env)

