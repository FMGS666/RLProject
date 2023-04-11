import gym
import numpy as np
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .board import Board

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "tictactoe_v3",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.board = Board()

        self.agents = ["player_1", "player_2"]
        self.possible_agents = self.agents[:]  #delete this

        self.action_spaces = {i: spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))) for i in self.agents}  
        self.observation_spaces = {   #since the action space is the same for both agents, there is no need to create a dictionary as we can just create one object without mapping
        # like this spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(3, 3, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}  #delete this
        self._cumulative_rewards = {i: 0 for i in self.agents}   #delete this
        self.terminations = {i: False for i in self.agents} #delete this
        self.truncations = {i: False for i in self.agents}  #delete this
        self.infos = {i: {"legal_moves": [(row, col) for row in range(3) for col in range(3)]} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.render_mode = render_mode


    def observe(self, agent):
        board_vals = self.board.squares
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)

        observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        legal_moves = self._legal_moves() if agent == self.agent_selection else []

        action_mask = np.zeros((3, 3), "int8")
        for row, col in legal_moves:
            action_mask[row, col] = 1

        return {"observation": observation, "action_mask": action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _legal_moves(self):
        return [(row, col) for row in range(3) for col in range(3) if self.board.squares[row, col] == 0]

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        row, col = action
        assert self.board.squares[row, col] == 0, "played illegal move"

        self.board.play_turn(self.agents.index(self.agent_selection), (row, col))

        next_agent = self._agent_selector.next()

        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                pass
            elif winner == 0:
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            else:
                self.rewards[self.agents[1]] += 1
                self.rewards[self.agents[0]] -= 1

            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.board = Board()

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        def get_symbol(input_value):
            if input_value == 0:
                return " "
            elif input_value == 1:
                return "X"
            else:
                return "O"

        board = np.array([get_symbol(val) for val in self.board.squares.flatten()]).reshape(3, 3)

        horizontal_line = '-' * 12
        for i in range(3):
            row = '|'.join([f' {board[i, j]} ' for j in range(3)])
            print(row)
            if i < 2:
                print(horizontal_line)



    def close(self):
        pass

