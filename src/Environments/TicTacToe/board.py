import numpy as np

class Board:
    def __init__(self):
        self.squares = np.zeros((3, 3), dtype=int)
        self.total_moves = 0
        self.winner = None

    def play_turn(self, agent, pos):
        row, col = pos
        if self.squares[row, col] != 0:
            return
        if agent == 0:
            self.squares[row, col] = 1
        elif agent == 1:
            self.squares[row, col] = 2
        self.total_moves += 1
        self.check_for_winner(agent, pos)

    def check_for_winner(self, agent, pos):
        if self.winner is not None:
            return

        row, col = pos
        player_mark = agent + 1
        row_marks = self.squares[row, :]
        col_marks = self.squares[:, col]
        diag_marks = self.squares.diagonal()
        anti_diag_marks = np.fliplr(self.squares).diagonal()

        win_conditions = [
            np.all(row_marks == player_mark),
            np.all(col_marks == player_mark),
            np.all(diag_marks == player_mark),
            np.all(anti_diag_marks == player_mark),
        ]
        if any(win_conditions):
            return True
        return False

    def check_game_over(self):
        if self.winner is not None:
            return True
        if self.total_moves == 9:
            return True
        return False

    def __str__(self):
        return str(self.squares)
