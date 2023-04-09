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
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

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

    def __update_board(
            self, 
            action: int
        ) -> None:
        """
        
        """
        piece = self.agents.index(self.agent_selection) + 1
        chosen_column = self.board[:, action]
        empty_rows = np.argwhere(chosen_column == 0)
        target_row_index = np.max(empty_rows)
        target_index = (target_row_index, action)
        self.board[target_index] = piece

    def step(
            self,
            action: int
        ) -> None:
        """
        
        """
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        assert self.legal_moves[action] == 1, "illegal move played"
        self.__update_board(action)
        self.__update_legal_moves(action)
        is_draw = self.check_for_draw()
        is_won = self.check_for_winner
        next_agent = self._agent_selector.next()
        if is_won:
            self.rewards[self.agent_selection] += 1
            self.rewards[next_agent] -= 1
            self.terminations = {i: True for i in self.agents}
        elif is_draw:
            self.terminations = {i: True for i in self.agents}
        else:
            self.agent_selection = next_agent
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

        
    def check_for_draw(
            self
        ) -> bool:
        """
        
        """
        return True if (np.all(self.legal_moves == 0) and not self.check_for_winner()) else False

    def reset(
            self
        ) -> None:
        """
        
        """
        self.__init__(
            self.grid_height, 
            self.grid_width, 
            self.render_mode        
        )

    def check_for_winner(
            self
        ) -> bool:
        piece = self.agents.index(self.agent_selection) + 1 # ??

        for c in range(self.grid_width - 3):
            for r in range(self.grid_height):
                if (
                    self.board[r][c] == piece
                    and self.board[r][c + 1] == piece
                    and self.board[r][c + 2] == piece
                    and self.board[r][c + 3] == piece
                ):
                    return True

        # Check vertical locations for win
        for c in range(self.grid_width):
            for r in range(self.grid_height - 3):
                if (
                    self.board[r][c] == piece
                    and self.board[r + 1][c] == piece
                    and self.board[r + 2][c] == piece
                    and self.board[r + 3][c] == piece
                ):
                    return True

        # Check positively sloped diagonals
        for c in range(self.grid_width - 3):
            for r in range(self.grid_height - 3):
                if (
                    self.board[r][c] == piece
                    and self.board[r + 1][c + 1] == piece
                    and self.board[r + 2][c + 2] == piece
                    and self.board[r + 3][c + 3] == piece
                ):
                    return True

        # Check negatively sloped diagonals
        for c in range(self.grid_width - 3):
            for r in range(3, self.grid_height):
                if (
                    self.board[r][c] == piece
                    and self.board[r - 1][c + 1] == piece
                    and self.board[r - 2][c + 2] == piece
                    and self.board[r - 3][c + 3] == piece
                ):
                    return True
        return False

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        screen_width = 1287
        screen_height = 1118
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.event.get()
        elif self.screen is None:
            self.screen = pygame.Surface((screen_width, screen_height))

        # Load and scale all of the necessary images
        tile_size = (screen_width * (91 / 99)) / 7

        red_chip = get_image(os.path.join("img", "C4RedPiece.png"))
        red_chip = pygame.transform.scale(
            red_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        black_chip = get_image(os.path.join("img", "C4BlackPiece.png"))
        black_chip = pygame.transform.scale(
            black_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13)))
        )

        board_img = get_image(os.path.join("img", "Connect4Board.png"))
        board_img = pygame.transform.scale(
            board_img, ((int(screen_width)), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # Blit the necessary chips and their positions
        for i in range(0, 42):
            if self.board[i] == 1:
                self.screen.blit(
                    red_chip,
                    (
                        (i % 7) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 7) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )
            elif self.board[i] == 2:
                self.screen.blit(
                    black_chip,
                    (
                        (i % 7) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / 7) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )

        if self.render_mode == "human":
            pygame.display.update()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )