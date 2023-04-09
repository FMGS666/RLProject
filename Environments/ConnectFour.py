import gymnasium as gym
import numpy as np

from gymnasium import spaces
from typing import Any
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


class ConnectFourEnvironment(AECEnv):
    """
    
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "connect_four_v3",
        "is_parallelizable": False,
        "render_fps": 2,
    }
    def __init__(
            self, 
            grid_height: int = 6, 
            grid_width: int = 7, 
            render_mode: Any = None
        ) -> None:
        """
        
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.render_mode = render_mode
        self.board_shape = (self.grid_height, self.grid_width)
        self.board = np.zeros(self.board_shape)
        self.action_space = spaces.Discrete(self.grid_width)
        self.agents = [f"agent{idx}" for idx in range(2)]
        self.action_spaces = spaces.Dict(
            {
                agent: self.action_space for agent in self.agents
            }
        )
        self.legal_moves = spaces.Box(
            low = 0, high = 1, shape = (self.grid_width, ), dtype = np.int8 
        )
        self.observation_space = spaces.Dict(
            {
                agent: {
                    "observation": spaces.Box(
                        low = 0, high = 1, shape = self.board_shape + (2, ), dtype = np.int8
                    )
                }
                for agent in self.agents
            }
        )
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observation_space(
            self, 
            agent: AgentID
        ) -> spaces.Box:
        """
        
        """
        return self.observation_spaces[agent]

    def action_space(
            self, 
            agent: AgentID
        ) -> spaces.Box:
        """
        
        """
        return self.action_spaces[agent]

    def observe(
            self, 
            agent_id: pettingzoo.AgentID
        ) -> dict[str, spaces.Box]:
        """
        
        """
        current_player = self.agents.index(agent_id)
        opponent_player = (current_player + 1) % 2
        current_player_moves = np.equal(self.board, current_player + 1)
        opponent_player_moves = np.equal(self.board, opponent_player + 1)
        observation = np.stack(
            [
                current_player_moves, 
                opponent_player_moves
            ], 
            axis=2
        ).astype(np.int8)
        return {"observation": observation, "action_mask": self.legal_moves}
    
     def __update_legal_moves(
            self,
            action: int
        ) -> None:
        """
    
        updates legal moves after each call to self.step()
    
        """
        legal_moves = np.argwhere(self.board[:, -1] == 0)
        for legal_move in legal_moves:
            self.legal_moves[legal_move] = 1
        
    def step(
            self,
            action: int
        ) -> None:
        """
        
        """
        ... # do stuff for the current step
        self.__update_legal_moves(action)

    def check_for_draw(
            self
        ) -> bool:
        """
        
        """
        return True if np.all(self.legal_moves == 0) else return False

    def reset(
            self
        ) -> None:
        """
        
        """       
        self.board = np.zeros(self.board_shape)
        ... # reset the stuff we need to reset

    def check_for_winner(
            self
        ) -> bool:
        piece = self.agents.index(self.agent_selection) + 1 # ??

        # Check horizontal locations for win
        column_count = 7
        row_count = 6

        for c in range(column_count - 3):
            for r in range(row_count):
                if (
                    self.board[r][c] == piece
                    and self.board[r][c + 1] == piece
                    and self.board[r][c + 2] == piece
                    and self.board[r][c + 3] == piece
                ):
                    return True

        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - 3):
                if (
                    self.board[r][c] == piece
                    and self.board[r + 1][c] == piece
                    and self.board[r + 2][c] == piece
                    and self.board[r + 3][c] == piece
                ):
                    return True

        # Check positively sloped diagonals
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if (
                    self.board[r][c] == piece
                    and self.board[r + 1][c + 1] == piece
                    and self.board[r + 2][c + 2] == piece
                    and self.board[r + 3][c + 3] == piece
                ):
                    return True

        # Check negatively sloped diagonals
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if (
                    self.board[r][c] == piece
                    and self.board[r - 1][c + 1] == piece
                    and self.board[r - 2][c + 2] == piece
                    and self.board[r - 3][c + 3] == piece
                ):
                    return True
        return False