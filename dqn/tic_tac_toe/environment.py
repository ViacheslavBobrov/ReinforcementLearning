import time
from enum import Enum

import numpy as np


class EpisodeStateCode(Enum):
    IN_PROGRESS = 1
    DRAW = 2
    WIN = 3


class TickTacToeEnvironment:
    player_sign_dict = {
        0: '_',
        1: 'X',
        2: '0'
    }

    def __init__(self):
        self.game_cells = np.zeros((3, 3), dtype=np.int32)

    def _check_win(self):
        for i in range(3):
            if self.game_cells[i, 0] == self.game_cells[i, 1] == self.game_cells[i, 2] and self.game_cells[i, 0] != 0:
                return True

        for j in range(3):
            if self.game_cells[0, j] == self.game_cells[1, j] == self.game_cells[2, j] and self.game_cells[0, j] != 0:
                return True

        if self.game_cells[0, 0] == self.game_cells[1, 1] == self.game_cells[2, 2] and self.game_cells[0, 0] != 0:
            return True
        if self.game_cells[0, 2] == self.game_cells[1, 1] == self.game_cells[2, 0] and self.game_cells[0, 2] != 0:
            return True
        return False

    def _set_cell(self, action, player_number):
        row = action // 3
        column = action % 3

        if self.game_cells[row, column] != 0:
            raise ValueError("Action can only be made for for a free cell, cell {0} is occupied".format(action + 1))
        self.game_cells[row, column] = player_number

    def _obtain_observation(self, player_number):
        game_cells_copy = self.game_cells.copy()
        if player_number == 2:
            for i in range(3):
                for j in range(3):
                    if game_cells_copy[i, j] == 1:
                        game_cells_copy[i, j] = 2
                    elif game_cells_copy[i, j] == 2:
                        game_cells_copy[i, j] = 1
        return game_cells_copy.ravel()

    def reset(self):
        self.game_cells = np.zeros((3, 3), dtype=np.int32)
        return self.game_cells.ravel()

    def step(self, action, action_player_id, observation_player_id):
        episode_state_code = EpisodeStateCode.IN_PROGRESS
        self._set_cell(action, action_player_id)
        if self._check_win():
            episode_state_code = EpisodeStateCode.WIN
        else:
            free_cells, _ = self.get_free_and_occupied_cells()
            if len(free_cells) == 0:
                episode_state_code = EpisodeStateCode.DRAW

        return self._obtain_observation(observation_player_id), episode_state_code

    def render(self):
        "Can be improved"
        print('\n' * 25)
        for i in range(3):
            for j in range(3):
                print(self.player_sign_dict[self.game_cells[i, j]] + ' ', end="")
                if j == 2:
                    print()
        time.sleep(0.8)
        print()

    def get_free_and_occupied_cells(self):
        free_cells = []
        occupied_cells = []
        for i in range(3):
            for j in range(3):
                cell_index_raveled = i * 3 + j
                if self.game_cells[i, j] == 0:
                    free_cells.append(cell_index_raveled)
                else:
                    occupied_cells.append(cell_index_raveled)
        return free_cells, occupied_cells
