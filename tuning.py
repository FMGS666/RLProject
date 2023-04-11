from src.Environments.Connect4 import *
from src.Agents.ExpectedSarsa import *
import pprint

env = Connect4Environment()


epsilon_grid = [
    1e-1,
    1e-2, 
    1e-3,
    2e-2,
    2e-2, 
    2e-3, 
]
gamma_grid = [
    1, 
    75e-2, 
    5e-1,
    25e-2
]
alpha_grid = [
    1, 
    5e-1, 
    1e-1, 
    5e-2, 
    1e-2, 
    5e-3, 
    1e-3
]

N_EPISODES = int(5e+4)
SEED = 1024


agents = []
for epsilon in epsilon_grid:
    for gamma in gamma_grid:
        for alpha in alpha_grid:
            agent = ExpectedSarsa(
                alpha = alpha, 
                epsilon = epsilon, 
                gamma = gamma,
                seed = SEED,
            )
            agents.append(agent)




from tqdm import tqdm
for agent in tqdm(agents):
    print("###############################################################")
    agent.train_n_episodes(env, N_EPISODES)

