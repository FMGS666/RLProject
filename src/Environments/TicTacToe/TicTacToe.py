import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .board import Board

class TicTacToeEnvironment(AECEnv):

    def __init__(self, render_mode=None):
        super().__init__()
        self.board = Board()

        self.agents = [f"agent{idx}" for idx in range(2)]

        self.action_spaces = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))) 
         
        self.observation_spaces =  spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=1, shape=(3, 3, 2), dtype=np.int8
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(3, 3), dtype=np.int8),
            }
        )
        self.legal_moves = np.ones((3, 3), dtype = np.int8)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.game_over = False
        self.winner = 0

    def observe(
            self, 
            agent: str
        ):
        board_vals = self.board.squares
        cur_player = self.agents.index(agent)
        opp_player = (cur_player + 1) % 2

        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)

        observation = np.stack(
            [
                cur_p_board, 
                opp_p_board
            ], 
            axis=2
        ).astype(np.int8)

        return {"observation": observation, "action_mask": self.legal_moves}

    def check_for_draw(
            self,
            cur_agent_idx: int, 
            row: int, 
            col: int
        ) -> bool:
        """
        
        """
        return True if (np.all(self.legal_moves == 0) and not self.board.check_for_winner(cur_agent_idx, (row, col))) else False

    def __update_legal_moves(
            self, 
            action: tuple()
        ) -> None:
        self.legal_moves[action] = 0

    def update_legal_moves(
            self, 
            action: tuple
        ) -> None:
        self.__update_legal_moves(action)

    def step(
            self, 
            action: tuple
        ) -> int:
        cur_agent_idx = self.agents.index(self.agent_selection)
        row, col = action
        if self.legal_moves[action] != 1:
            self.game_over = True
            return -100
        self.__update_legal_moves(action)
        is_draw = self.check_for_draw(cur_agent_idx, row, col)
        is_won = self.board.check_for_winner(cur_agent_idx, (row, col))
        self.board.play_turn(cur_agent_idx, (row, col))
        if is_won:
            self.game_over = True
            self.winner = self.agents.index(self.agent_selection) + 1
            return 100
        elif is_draw:
            self.game_over = True
            return 0
        else:
            next_agent = self._agent_selector.next()
            return 1

    def reset(
            self, 
        ) -> None:
        self.__init__()


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
