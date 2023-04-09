from __future__ import annotations

from tqdm import tqdm 
from typing import Callable

import os, sys, re, json
import gymnasium as gym
import time
import text_flappy_bird_gym
import numpy as np


class Agent(object):
    def __init__(
            self, 
            width: int, 
            height: int, 
            n_actions: int = 2, 
            epsilon: float = 1e-1,
            alpha: float = 5e-1,
            gamma: float = 1e-0, 
            n_steps: int = 1, 
            debug: int = 0, 
            seed: int = 236 ,
            verbose: int = 1, 
            name: str = "GenericRlAgent"
        ) -> None:
        
        # initializing the attributes
        self._width = width
        self._height = height
        self._n_actions = n_actions
        self.epsilon = epsilon
        self._debug = debug
        self._seed = seed
        self.verbose = verbose
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.n_steps = n_steps

        # initializing dimensions for the action-state matrix
        self._n_possible_vertical_distances = self._height * 2
        self._n_possible_horizontal_distances = self._width
        self._n_possible_heights = self._height
        self._action_state_target_dimension = (
            self._n_possible_heights, 
            self._n_possible_vertical_distances, 
            self._n_possible_horizontal_distances, 
            self._n_actions
        ) if self.consider_height \
            else (
                self._n_possible_vertical_distances, 
                self._n_possible_horizontal_distances, 
                self._n_actions
            )
        
        ## MODIFY THIS FOR THE CURRENT USE
        # initializing the action-state matrix
        self.action_state_value_matrix = np.zeros(
            self._action_state_target_dimension
        )

        ## MODIFY THIS FOR THE CURRENT USE
        # initializing the action-state counter matrix
        self.action_state_counter_matrix = np.zeros(
            self._action_state_target_dimension
        )

        # initializing current score
        self.current_score = 0
        self.max_score = 0

        # initialize actions type counter
        self.n_greedy_actions = 0
        self.n_exploratory_actions = 0

        # initializing random numbers generator
        self._random_generator = np.random.RandomState(seed)


    ## MODIFY THIS FOR THE CURRENT USE
    def __find_state_index(
            self, 
            obs: tuple[int],
        ) -> tuple:
        horizontal_distance, vertical_distance = obs
        _, height = info[self._info_field]
        return (height - 1, vertical_distance, horizontal_distance) if self.consider_height \
            else (vertical_distance, horizontal_distance)

    ## MODIFY THIS FOR THE CURRENT USE
    def __find_action_state_index(
            self, 
            obs: tuple[int],
            action: int, 
        ) -> tuple:
        if self.consider_height:
            height, vertical_distance, horizontal_distance = self.__find_state_index(
                obs,
            )
        else:
            vertical_distance, horizontal_distance = self.__find_state_index(
                obs, 
            ) 
        return (height - 1, vertical_distance, horizontal_distance, action) if self.consider_height \
            else (vertical_distance, horizontal_distance, action)

    ## MODIFY THIS FOR THE CURRENT USE
    def _get_state_value(
            self,
            obs: tuple[int],
        ) -> np.ndarray:
        state_index = self.__find_state_index(
            obs, 
        )
        if self._debug > 1:
            print(f"state_index {state_index}")
        state_value_matrix = self.action_state_value_matrix[state_index]
        return state_value_matrix

    ## MODIFY THIS FOR THE CURRENT USE
    def _get_action_state_value(
            self,
            obs: tuple[int], 
            action: int,             
        ) -> float:
        action_state_index = self.__find_action_state_index(
            obs,
            action,  
        )
        action_state_value = self.action_state_value_matrix[action_state_index]
        return action_state_value

    ## MODIFY THIS FOR THE CURRENT USE
    def _set_action_state_value(
            self,
            obs: tuple[int],
            action: int,
            value: float,              
        ) -> None:
        action_state_index = self.__find_action_state_index(
            obs,
            action,  
        )
        self.action_state_value_matrix[action_state_index] = value

    ## MODIFY THIS FOR THE CURRENT USE
    def _increment_counter_action_state(
            self,
            obs: tuple[int],
            action: int,             
        ) -> None:
        action_state_index = self.__find_action_state_index(
            obs,
            action, 
        )
        self.action_state_counter_matrix[action_state_index] +=1
    
    def __policy(
            self, 
            obs: tuple, 
            count_action_type: bool = False, 
        ) -> int:
        state_value = self._get_state_value(
            obs,
        )
        if self._debug:    
            assert state_value.shape[0] == self._n_actions and len(state_value.shape) == 1, \
                f"The action_values array has an incorrect shape of {state_value.shape}"
        greedy = self._random_generator.random() >= self.epsilon
        if greedy:
            if count_action_type:
                self.n_greedy_actions += 1
            best_action = state_value.argmax()
            return best_action
        else:
            action = np.random.choice([idx for idx, element in enumerate(state_value)])
            if count_action_type:
                self.n_exploratory_actions += 1
            return action
    
    def _policy(
            self,
            obs: tuple, 
            count_action_type: bool = False, 
        ) -> int:
        return self.__policy(
            obs, 
            info, 
            count_action_type, 
        )
    
    def __generate_description_string(
            self
        ) -> str:
        description_string = f"{self.name}_maxScore{self.max_score}_w{self._width}_h{self._height}_nAct{self._n_actions}_eps{self.epsilon}_nSteps{self.n_steps}_gamma{self.gamma}_alpha{self.alpha}"
        return description_string

    def generate_description_string():
        return self.__generate_description_string()

    def _save_history(
            self, 
            episodes_lengths: list[int], 
            episodes_scores: list[int],
            path_name: str | Path = ".\ESHistory"
        ) -> None:
        description_string = self.__generate_description_string() + ".json"
        file_name = os.path.join(path_name, description_string)
        dump_dictionary = {
            "episodes_lengths": episodes_lengths, 
            "episodes_scores": episodes_scores
        }
        with open(file_name, "w") as file_dump:
            json.dump(dump_dictionary, file_dump)

    def _dump(
            self,
            path_name: str | Path = ".\ESAgents"
        ) -> None:
        description_string = self.__generate_description_string() + ".npy"
        file_name = os.path.join(path_name, description_string)
        assert re.match(".*\.npy", file_name), \
            "FileNameError: file_name provided is in the wrong format, please save array as .npy"
        if self.verbose:
            print(f"\nDumping agent:\n\tdescription_string -> {description_string}\n\n\tpath_name -> {path_name}\n\n\tfilename -> {file_name}")
        np.save(file_name, self.action_state_value_matrix)

    def load(
            self, 
            filename: str | Path
        ) -> None:
        action_value_matrix = np.load(filename)
        self.action_state_value_matrix = action_value_matrix 