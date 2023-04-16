from src.Environments.Connect4.Connect4 import *
from src.Environments.TicTacToe.TicTacToe import *
from src.Agents.ExpectedSarsa import *
import pprint

env = Connect4Environment()

agent = ExpectedSarsa()

n=100
wins=0
draws=0
losses=0

def score_performance(wins, losses, draws):
    return wins*5 + draws * 1 - losses *5

import matplotlib.pyplot as plt

"""
def find_path_with_best_params(agent, env, n):
    eps_values = [0.001, 0.01, 0.1, 0.5]
    gamma_values = [0.25, 0.5, 0.75, 1]
    alpha_values = [0.1, 0.5, 1]

    scores = []
    params = []

    for eps in eps_values:
        for gamma in gamma_values:
            for alpha in alpha_values:
                wins = 0
                draws = 0
                losses = 0
                file_name = f'TrainedAgents/ExpectedSarsaConnect4/ExpectedSarsa_eps{eps}_gamma{gamma}_alpha{alpha}.json'
                agent.load(file_name)
                for i in range(n):
                    result = agent.play_against_random(env)
                    if result == 1:
                        wins += 1
                    elif result == 2:
                        losses += 1
                    else:
                        draws += 1

                score = score_performance(wins, losses, draws)
                print(score, f" with : eps :{eps}, gamma :{gamma}, alpha :{alpha}")
                scores.append(score)
                params.append((eps, gamma, alpha))

    return scores, params

def find_path_with_best_params_eps(agent, env, n):
    eps_values = [0.001, 0.01, 0.1, 0.5]
    gamma_values = [0.25, 0.5, 0.75, 1]
    alpha_values = [0.1, 0.5, 1]

    results = {}

    for eps in eps_values:
        scores = []
        params = []

        for gamma in gamma_values:
            for alpha in alpha_values:
                wins = 0
                draws = 0
                losses = 0
                file_name = f'TrainedAgents/ExpectedSarsaConnect4/ExpectedSarsa_eps{eps}_gamma{gamma}_alpha{alpha}.json'
                agent.load(file_name)
                for i in range(n):
                    result = agent.play_against_random(env)
                    if result == 1:
                        wins += 1
                    elif result == 2:
                        losses += 1
                    else:
                        draws += 1

                score = score_performance(wins, losses, draws)
                print(score, f" with : eps :{eps}, gamma :{gamma}, alpha :{alpha}")
                scores.append(score)
                params.append((gamma, alpha))

        results[eps] = (scores, params)

    return results
"""
def find_path_with_best_params(agent, env, n, param_to_fix):
    eps_values = [0.001, 0.01, 0.1, 0.5]
    gamma_values = [0.25, 0.5, 0.75, 1]
    alpha_values = [0.1, 0.5, 1]

    results = {}

    if param_to_fix not in ["eps", "gamma", "alpha"]:
        raise ValueError("Invalid param_to_fix value. Must be one of: 'eps', 'gamma', 'alpha'")

    for param in (eps_values if param_to_fix == "eps" else gamma_values if param_to_fix == "gamma" else alpha_values):
        param_results = []
        for gamma in gamma_values:
            for alpha in alpha_values:
                if param_to_fix == "eps":
                    eps = param
                elif param_to_fix == "gamma":
                    eps = eps_values[1]
                    gamma = param
                else:
                    eps = eps_values[1]
                    gamma = gamma_values[1]
                    alpha = param

                wins = 0
                draws = 0
                losses = 0
                file_name = f'TrainedAgents/ExpectedSarsaConnect4/ExpectedSarsa_eps{eps}_gamma{gamma}_alpha{alpha}.json'
                agent.load(file_name)
                for i in range(n):
                    result = agent.play_against_random(env)
                    if result == 1:
                        wins += 1
                    elif result == 2:
                        losses += 1
                    else:
                        draws += 1

                score = score_performance(wins, losses, draws)
                print(score, f" with : eps :{eps}, gamma :{gamma}, alpha :{alpha}")
                param_results.append((score, (eps, gamma, alpha)))
        results[param] = param_results

    return results



import matplotlib.pyplot as plt

def plot_scores(results, param_to_fix):
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'm']

    for idx, (param_value, param_results) in enumerate(results.items()):
        x_labels = [f"({gamma}, {alpha})" for _, (_, gamma, alpha) in param_results]
        scores = [score for score, _ in param_results]
        x = range(len(scores))

        ax.plot(x, scores, marker='o', color=colors[idx], label=f'{param_to_fix}={param_value}')

        for i, (eps, gamma, alpha) in enumerate([params for _, params in param_results]):
            ax.annotate(f"({eps}, {gamma}, {alpha})", (x[i], scores[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(f'Gamma, Alpha Combinations (Fixed {param_to_fix})')
    ax.set_ylabel('Score')
    ax.set_title(f'Evolution of the Score for Each Value of {param_to_fix}')
    ax.legend()
    plt.show()  


#results = find_path_with_best_params(agent, env, n, param_to_fix="gamma")
#plot_scores(results, param_to_fix="gamma")  # or "alpha" or "gamma"


while(wins<440):
    wins=0
    losses=0 
    for i in range(1000):
        agent.load('TrainedAgents/ExpectedSarsaConnect4/ExpectedSarsa_eps0.01_gamma0.25_alpha0.1.json')
        result= agent.play_against_random(env)
        if result==1:
            wins+=1
        elif result==2:
            losses+=1
    print(wins)
    
        

    
print("wins: ", wins)
print("losses: ", losses)
print("draws : ",1000-wins-losses )
