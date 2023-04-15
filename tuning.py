from src.Environments.Connect4 import *
from src.Agents.ExpectedSarsa import *
import pprint
from tqdm import tqdm


def save_checkpoint(
        idx: int,
        filename: str = ".\checkpoint.txt"
    ) -> None:
    with open(filename, "w") as file_handle:
        file_handle.write(str(idx + 1))

N_EPISODES = int(1e+4)
SEED = 1024

env = Connect4Environment()


epsilon_grid = [
    5e-1,
    1e-1,
    1e-2, 
    1e-3,
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
    1e-1
]

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

print(len(agents))
for idx, agent in tqdm(enumerate(agents[1:])):
    print("###############################################################")
    agent.train_n_episodes(env, N_EPISODES)
    save_checkpoint(idx)