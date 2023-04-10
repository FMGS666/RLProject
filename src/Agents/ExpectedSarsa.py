from Agents.Agent import * 


class ExpectedSarsa(Agent):
    def __init__(
            self, 
            grid_width: int = 7, 
            grid_height: int = 6, 
            epsilon: float = 1e-1,
            alpha: float = 5e-1, 
            gamma: float = 1e-0, 
            n_steps: int = 1, 
            debug: int = 0, 
            seed: int = 236,
            verbose: int = 1,
            name: str = "ExpectedSarsa",
            render: bool = False
        ) -> None:
        super(ExpectedSarsa, self).__init__(
            grid_width, 
            grid_height, 
            epsilon = epsilon, 
            debug = debug,
            seed = seed, 
            verbose = verbose,
            name = name,
            gamma = gamma, 
            alpha = alpha, 
            n_steps = n_steps
        )
        self.render = render

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
            reward
        ) -> float:
        state_value = self._get_state_value(
            obs1, 
        )
        greedy_action = np.argmax(state_value)
        policy_distribution = np.full(shape = self._n_actions, fill_value = self.epsilon/self._n_actions)
        policy_distribution[greedy_action] += (1 - self.epsilon)
        action_state_value = np.dot(state_value.T, policy_distribution)
        sarsa_value = reward + self.gamma * action_state_value
        return sarsa_value
    
    def __compute_update_value(
            self, 
            action: int, 
            obs: np.ndarray, 
            obs1: np.ndarray,
            reward: float, 
            done: bool,
        ) -> float:
        if self._debug:
            print(f"\n\n__compute_update_value\n\tobs {obs} info {info}, done {done}\n\ttype(obs) {type(obs)} type(info) {type(info)} type(done) {type(done)}")
        if done:
            if self._debug: 
                print(f"done == {done}, returning 0 as action-state value")
            return 0
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
            obs: tuple, 
            obs1: tuple, 
            reward: float,
            env: gym.Env, 
            done: bool
        ) -> None:
        if self._debug:
            print(f"\n__update_action_state_value\n\tobs {obs} info {info}, {type(obs)} type(info) {type(info)}")
        update_value = self.__compute_update_value(
            action, 
            obs, 
            obs1, 
            reward, 
            done
        )
        if self._debug > 1:
            print(f"\n\nupdate_value {update_value} for obs {obs}, info {info[self._info_field][-1]}")
        self._set_action_state_value(
            obs, 
            action, 
            update_value, 
        )

    def __train_one_episode_against_itself(
            self,
            env: AECEnv, 
        ) -> None:
        ...

    def __print_training_description_message(
            self, 
            n_episodes: int, 
            episodes_scores: list,
            begin: bool
        ) -> None:
        message = f"""
        
        Training TemporalDifferenceAgent for:

            n_episodes -> {n_episodes}

        Parameters of the environment:

            grid_width ->  {self.grid_width}
            grid_height -> {self.grid_height}

        Parameters of the agent: 

            alpha (step_size) -> {self.alpha}
            gamma (discount_factor) -> {self.gamma}
            epsilon (exploration_rate) -> {self.epsilon}
            n_steps -> {self.n_steps}

        """ if begin else \
        f"""
        
        Trained for {n_episodes}:

            best score -> {self.max_score}
            average score -> {np.mean(episodes_scores)}
            std -> {np.std(episodes_scores)}


        """
        print(message)

    def __train_n_episodes(
            self, 
            env: AECEnv, 
            n_episodes: int, 
            patience: int = 1e+6,
            dump: bool = True 
        ) -> None:
        if self.verbose:
            self.__print_training_description_message(
                n_episodes, 
                episodes_scores, 
                True
            )
        for episode in tqdm(range(n_episodes)):
            self.__train_one_episode(env)
            if episode % 10000 == 0 and self.verbose > 1:
                print("\rEpisode {}/{}, Score: {}, Max score: {}".format(episode, n_episodes, episodes_scores[-1], self.max_score), end="")
                sys.stdout.flush()
        if self.verbose:
            self.__print_training_description_message(
                n_episodes, 
                episodes_scores, 
                False
            )
        if self.max_score and dump:
            self._dump()
            self._save_history(episodes_lengths, episodes_scores)

    def train_n_episodes(
            self, 
            env: gym.Env, 
            n_episodes: int,
            patience: int = 1e+6
        ) -> None:
        return self.__train_n_episodes(env, n_episodes)