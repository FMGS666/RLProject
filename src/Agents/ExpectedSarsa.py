from .Agent import * 


class ExpectedSarsa(Agent):
    def __init__(
            self, 
            grid_width: int = 7, 
            epsilon: float = 1e-1,
            alpha: float = 5e-1, 
            gamma: float = 1e-0, 
            debug: int = 0, 
            seed: int = 236,
            verbose: int = 1,
            name: str = "ExpectedSarsa",
        ) -> None:
        super(ExpectedSarsa, self).__init__(
            grid_width, 
            epsilon = epsilon, 
            debug = debug,
            seed = seed, 
            verbose = verbose,
            name = name,
            gamma = gamma, 
            alpha = alpha, 
        )

    def policy(
            self, 
            obs: np.ndarray,
            count_action_type: bool = False, 
        ) -> int:
        return super(ExpectedSarsa, self)._policy(
            obs,
            count_action_type = count_action_type, 
        )
    
    def __compute_step_value(
            self, 
            obs1: np.ndarray,
            reward: float | int
        ) -> float:
        state_value = self._get_state_value(
            obs1, 
        )
        greedy_action = state_value.argmax() if isinstance(self.grid_width, int)\
            else np.unravel_index(state_value.argmax(), state_value.shape)
        policy_distribution = np.full(shape = self.grid_width, fill_value = self.epsilon/self.grid_width) if isinstance(self.grid_width, int)\
            else np.full(shape = self.grid_width, fill_value = self.epsilon/len(self.grid_width))
        policy_distribution[greedy_action] += (1 - self.epsilon)
        action_state_value = np.dot(state_value.T, policy_distribution) if isinstance(self.grid_width, int)\
            else np.dot(state_value.flatten().T, policy_distribution.flatten())
        sarsa_value = reward + self.gamma * action_state_value
        return sarsa_value
    
    def __compute_update_value(
            self, 
            action: int, 
            obs: np.ndarray, 
            obs1: np.ndarray,
            reward: float, 
        ) -> float:
        current_action_state_value = self._get_action_state_value(
            obs, 
            action
        )
        n_step_value = self.__compute_step_value(
            obs1, 
            reward
        )
        update_value = current_action_state_value + \
            self.alpha * (n_step_value - current_action_state_value)
        return update_value
    
    def __update_action_state_value(
            self,
            action: int, 
            obs: np.ndarray, 
            obs1: np.ndarray, 
            reward: float, 
        ) -> None:
        update_value = self.__compute_update_value(
            action, 
            obs, 
            obs1, 
            reward, 
        )
        if self.debug > 1:
            print(f"\n\nupdate_value {update_value} for obs {obs}")
        self._set_action_state_value(
            obs,
            action, 
            update_value,
        )

    def __train_one_episode_against_itself(
            self,
            env: AECEnv, 
        ) -> None:
        env.reset()
        idx = 0
        agent_to_play = env.agents[idx]
        obs, legal_moves = env.observe(agent_to_play).values()  
        while not env.game_over:
            agent_to_play = env.agents[idx]
            obs1, legal_moves = env.observe(agent_to_play).values()
            action = self.policy(obs)
            reward = env.step(action)
            self.__update_action_state_value(action, obs, obs1, reward)
            idx = (idx + 1) % 2
            obs = obs1
            self.action_counts[action] += 1
        self.winners_history.append(env.winner)

    def play_against_random(
            self,
            env: AECEnv,
            target_idx: int = 0
        ) -> None:
        env.reset()
        idx = 0
        agent_to_play = env.agents[idx]
        obs, legal_moves = env.observe(agent_to_play).values()  
        while not env.game_over:
            agent_to_play = env.agents[idx]
            obs1, legal_moves = env.observe(agent_to_play).values()
            action = self.policy(obs) if idx==target_idx else np.random.choice([idx for idx in range(self.grid_width)]) 
            reward = env.step(action)
            idx = (idx + 1) % 2
            obs = obs1
        return env.winner

    def __print_training_description_message(
            self, 
            n_episodes: int, 
        ) -> None:
        message = f"""
        _________________________________________

        Training TemporalDifferenceAgent for:

            n_episodes -> {n_episodes}

        Parameters of the environment:

            grid_width ->  {self.grid_width}

        Parameters of the agent: 

            alpha (step_size) -> {self.alpha}
            gamma (discount_factor) -> {self.gamma}
            epsilon (exploration_rate) -> {self.epsilon}

        ___________________________________________

        """
        print(message)

    def train_n_episodes(
            self, 
            env: AECEnv, 
            n_episodes: int, 
            patience: int = 1e+6,
            dump: bool = True 
        ) -> None:
        if self.verbose:
            self.__print_training_description_message(n_episodes)
        for episode in tqdm(range(n_episodes)):
            self.__train_one_episode_against_itself(env)
            if self.verbose > 1:
                print("\rEpisode {}/{}, action_counts: {}".format(episode, n_episodes, self.action_counts), end="")
                sys.stdout.flush()
        if dump:
            self._dump(".\TrainedAgents\ExpectedSarsaTicTacToe")
            

    def load(
            self, 
            filename: str
        ) -> None:
        super(ExpectedSarsa, self)._load(filename)