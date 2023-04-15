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
            grid_width: tuple | int = 7, 
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
        self.epsilon = epsilon
        self.debug = debug
        self.seed = seed
        self.verbose = verbose
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        
        # initializing the action-state matrix
        self.action_state_value_dictionary = defaultdict(np.ndarray)

        # initialize actions type counter
        self.n_greedy_actions = 0
        self.n_exploratory_actions = 0

        # initializing random numbers generator
        self._random_generator = np.random.RandomState(seed)
        self.action_counts = np.zeros((self.grid_width, )) if isinstance(self.grid_width, int)\
            else np.zeros(self.grid_width)
        self.winners_history = []

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
        observed_states = dict(self.action_state_value_dictionary).keys()
        if encoded_observation in observed_states:
            state_values = self.action_state_value_dictionary[encoded_observation]
        else:
            state_values = np.zeros((self.grid_width, ), dtype = np.float32) if isinstance(self.grid_width, int)\
                else np.zeros(self.grid_width)
            self.action_state_value_dictionary[encoded_observation] = state_values
        return state_values

    def _get_action_state_value(
            self,
            obs: np.ndarray, 
            action: int | tuple,             
        ) -> float:
        state_value = self._get_state_value(
            obs
        )
        action_state_value = state_value[action]
        return action_state_value

    def _set_action_state_value(
            self,
            obs: np.ndarray,
            action: int | tuple,
            value: float,              
        ) -> None:
        encoded_observation = self.__encode_observation(
            obs
        )
        observed_states = dict(self.action_state_value_dictionary).keys()
        if encoded_observation in observed_states:
            self.action_state_value_dictionary[encoded_observation][action] = value
        else:
            state_values = np.zeros((self.grid_width, ), dtype = np.float32) if isinstance(self.grid_width, int)\
                else np.zeros(self.grid_width, dtype = np.float32)
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
        ) -> int | np.ndarray:
        state_value = self._get_state_value(
            obs,
        )
        greedy = self._random_generator.random() >= self.epsilon
        if greedy and not np.all(state_value == 0):
            if count_action_type:
                self.n_greedy_actions += 1
            best_action = state_value.argmax() if isinstance(self.grid_width, int)\
                else tuple(np.unravel_index(state_value.argmax(), state_value.shape))
            print(f"action {best_action}")            
            return best_action
        else:
            action = np.random.choice([idx for idx, element in enumerate(state_value)]) if isinstance(self.grid_width, int)\
                else tuple(np.random.randint(0, size) for size in state_value.shape)
            if count_action_type:
                self.n_exploratory_actions += 1
            print(f"action {action}")            
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
        description_string = f"{self.name}_eps{self.epsilon}_gamma{self.gamma}_alpha{self.alpha}"
        return description_string

    def generate_description_string():
        return self.__generate_description_string()

    def __save_winners_history(
            self,
            path_name: str | Path = ".\TrainedAgents"
        ) -> None:
        description_string = self.__generate_description_string() + "_winner_history.json"
        file_name = os.path.join(path_name, description_string)
        dump_dictionary = {
            "winners_history": self.winners_history,
        }
        with open(file_name, "w") as file_dump:
            json.dump(dump_dictionary, file_dump)

    @staticmethod
    def __array_to_list(array):
        result = []
        for row in array:
            sublist = []
            for element in row:
                sublist.append(float(element))
            result.append(sublist)
        return result


    def _dump(
            self,
            path_name: str | Path = ".\TrainedAgents"
        ) -> None:
        description_string = self.__generate_description_string() + ".json"
        file_name = os.path.join(path_name, description_string)
        assert re.match(".*\.json", file_name), \
            "FileNameError: file_name provided is in the wrong format, please save array as .npy"
        if self.verbose:
            print(f"\nDumping agent:\n\tdescription_string -> {description_string}\n\n\tpath_name -> {path_name}\n\n\tfilename -> {file_name}")
        action_state_value_dictionary = {key: self.__array_to_list(value) for key, value in self.action_state_value_dictionary.items()}
        with open(file_name, "w") as file_handle:
            json.dump(action_state_value_dictionary, file_handle)
        self.__save_winners_history(path_name = path_name)

    def _load(
            self, 
            filename: str
        ) -> None:
        with open(filename, "r") as file_handle:
            json_data = json.load(file_handle)
        self.action_state_value_dictionary = defaultdict(json_data)