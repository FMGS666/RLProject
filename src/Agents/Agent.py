from __future__ import annotations

from tqdm import tqdm 
from typing import Callable
from collections import defaultdict

import os, sys, re, json, time
import gymnasium as gym

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import time
import numpy as np


class Agent(object):
    def __init__(
            self, 
            grid_width: int = 7, 
            grid_height: int = 6,
            epsilon: float = 1e-1,
            alpha: float = 5e-1,
            gamma: float = 1e-0, 
            debug: int = 0, 
            seed: int = 236 ,
            verbose: int = 1, 
            name: str = "GenericRlAgent"
        ) -> None:
        
        # initializing the attributes
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.epsilon = epsilon
        self.debug = debug
        self.seed = seed
        self.verbose = verbose
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        
        # initializing the action-state matrix
        self.action_state_value_dictionary = defaultdict(np.ndarray)

        self.observed_states = dict(self.action_state_value_dictionary).keys()

        # initializing current score
        self.current_score = 0
        self.max_score = 0

        # initialize actions type counter
        self.n_greedy_actions = 0
        self.n_exploratory_actions = 0

        # initializing random numbers generator
        self._random_generator = np.random.RandomState(seed)
        self.action_counts = np.zeros((self.grid_width, ))

    @staticmethod
    def __encode_observation(
            obs: np.ndarray,
            axis: int = 1
        ) -> str:
        """
        
        encodes the board in a useful representation for the state    
        
        """
        assert len(obs.shape) == 3
        packed_observation = np.packbits(obs, axis = axis).flatten()
        packed_observation = np.char.mod('%d', packed_observation)
        return "".join(packed_observation)

    def _get_state_value(
            self,
            obs: np.ndarray,
        ) -> np.ndarray:
        encoded_observation = self.__encode_observation(
            obs
        )
        if encoded_observation in self.observed_states:
            state_values = self.action_state_value_dictionary[encoded_observation]
        else:
            state_values = np.zeros((self.grid_width, ), dtype = np.float32)
            self.action_state_value_dictionary[encoded_observation] = state_values
        assert (
            state_values.shape[0] == self.grid_width and 
            len(state_values.shape) == 1
        ), f"state value matrix has wrong shape of {state_values.shape}"
        return state_values

    def _get_action_state_value(
            self,
            obs: np.ndarray, 
            action: int,             
        ) -> float:
        state_value = self._get_state_value(
            obs
        )
        action_state_value = state_value[action]
        return action_state_value

    def _set_action_state_value(
            self,
            obs: np.ndarray,
            action: int,
            value: float,              
        ) -> None:
        encoded_observation = self.__encode_observation(
            obs
        )
        if encoded_observation in self.observed_states:
            self.action_state_value_dictionary[encoded_observation][action] = value
        else:
            state_values = np.zeros((self.grid_width, ), dtype = np.float32)
            state_values[action] = value
            self.action_state_value_dictionary[encoded_observation] = state_values

    def _increment_counter_action_state(
            self,
            obs: np.ndarray,
            action: int,             
        ) -> None:
        ...


    def __policy(
            self, 
            obs: np.ndarray, 
            count_action_type: bool = False, 
        ) -> int:
        state_value = self._get_state_value(
            obs,
        )
        if self.debug:    
            assert state_value.shape[0] == self._n_actions and len(state_value.shape) == 1, \
                f"The action_values array has an incorrect shape of {state_value.shape}"
        greedy = self._random_generator.random() >= self.epsilon
        if greedy and not np.all(state_value == 0):
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
            count_action_type, 
        )
    
    def __generate_description_string(
            self
        ) -> str:
        description_string = f"{self.name}_maxScore{self.max_score}_nAct{self._n_actions}_eps{self.epsilon}_gamma{self.gamma}_alpha{self.alpha}"
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