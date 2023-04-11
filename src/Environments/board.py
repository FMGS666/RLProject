import numpy as np

class Board:
    def __init__(self):
        self.squares = np.zeros((3, 3), dtype=int)

        self.calculate_winners()

    def setup(self):
        self.calculate_winners()

    def play_turn(self, agent, pos):
        row, col = pos
        if self.squares[row, col] != 0:
            return
        if agent == 0:
            self.squares[row, col] = 1
        elif agent == 1:
            self.squares[row, col] = 2
        return

    def calculate_winners(self):
        winning_combinations = []

        # Vertical combinations
        for col in range(3):
            winning_combinations.append([(row, col) for row in range(3)])

        # Horizontal combinations
        for row in range(3):
            winning_combinations.append([(row, col) for col in range(3)])

        # Diagonal combinations
        winning_combinations.append([(i, i) for i in range(3)])
        winning_combinations.append([(i, 2 - i) for i in range(3)])

        self.winning_combinations = winning_combinations

    def check_for_winner(self):
        winner = -1
        for combination in self.winning_combinations:
            states = [self.squares[row, col] for row, col in combination]
            if all(x == 1 for x in states):
                winner = 0
            if all(x == 2 for x in states):
                winner = 1
        return winner

    def check_game_over(self):
        winner = self.check_for_winner()

        if winner == -1 and np.all(self.squares != 0):
            return True
        elif winner in [0, 1]:
            return True
        else:
            return False

    def __str__(self):
        return str(self.squares)

