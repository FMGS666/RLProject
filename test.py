from src.Environments.Connect4 import *
from src.Agents.ExpectedSarsa import *

env = Connect4Environment()

env.step(2)
env.step(2)

obs = env.observe("agent0")

a = obs["observation"]
