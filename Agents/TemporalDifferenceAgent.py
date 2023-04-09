from Agents1.Agent import * 

class TemporalDifferenceAgent(Agent):
    def __init__(
            self, 
            width: int, 
            height: int, 
            n_actions: int = 2,
            alpha: float = 5e-1, 
            epsilon: float = 1e-1,
            gamma: float = 1e-0, 
            n_steps: int = 1, 
            debug: int = 0, 
            prior: bool = False, 
            info_field: str = "player", 
            score_field: str = "score", 
            distance_field: str = "distance",
            seed: int = 236,
            verbose: int = 1,
            consider_height: bool = True,
            name: str = "TDAgent"
        ) -> None:
        super(TemporalDifferenceAgent, self).__init__(
            width, 
            height, 
            n_actions, 
            epsilon = epsilon, 
            debug = debug, 
            prior = prior, 
            info_field = info_field, 
            score_field = score_field,  
            distance_field = distance_field,
            seed = seed, 
            verbose = verbose,
            consider_height = consider_height,
            name = name,
            gamma = gamma, 
            alpha = alpha, 
            n_steps = n_steps
        )

    def policy(
            self, 
            obs: tuple, 
            info: dict, 
            count_action_type: bool = False, 
        ) -> int:
        return super(TemporalDifferenceAgent, self)._policy(
            obs, 
            info, 
            count_action_type = count_action_type, 
        )

    def __compute_n_step_value(
            self, 
            obs1: tuple, 
            info1: dict, 
            reward: float, 
            env: gym.Env
        ) -> float:
        """
        Probably the source of all my problems
        """
        env_copy = env
        values = []
        current_obs = obs1; current_info = info1; current_reward = reward
        for step in range(self.n_steps):
            if step == 0:
                values.append(reward)
                action = self.policy(
                    current_obs, 
                    current_info
                )
                current_obs, current_reward, done, _, current_info = env_copy.step(action)
                continue
            discount = self.gamma ** step
            values.append(discount * current_reward)
            action = self.policy(
                current_obs, 
                current_info
            )
            if step != (self.n_steps - 1):
                current_obs, current_reward, done, _, current_info = env_copy.step(action)    
        discount = self.gamma ** self.n_steps
        value = super(TemporalDifferenceAgent, self)._get_action_state_value(
            current_obs,
            current_info,  
            action, 
        )
        values.append(discount * value)
        if self._debug:
            print(f"values {values}\nnp.sum(values) {np.sum(values)}")
        return np.sum(values)

    def __compute_step_value(
            self, 
            obs1: tuple[int], 
            info1: dict,
            reward
        ) -> float:
        action = self.policy(obs1, info1)
        action_state_value = self._get_action_state_value(
            obs1, 
            info1, 
            action
        )
        td_value = reward + self.gamma * action_state_value
        return td_value

    def __compute_update_value(
            self, 
            action: int, 
            obs: tuple, 
            obs1: tuple, 
            info: dict, 
            info1: dict, 
            reward: float, 
            done: bool,
            env: gym.Env
        ) -> float:
        if self._debug:
            print(f"\n\n__compute_update_value\n\tobs {obs} info {info}, done {done}\n\ttype(obs) {type(obs)} type(info) {type(info)} type(done) {type(done)}")
        if done:
            if self._debug: 
                print(f"done == {done}, returning 0 as action-state value")
            return 0
        current_action_state_value = self._get_action_state_value(
            obs, 
            info, 
            action
        )
        n_step_value = self.__compute_n_step_value(
            obs1, 
            info1, 
            reward, 
            env
        ) if self.n_steps > 1 \
            else self.__compute_step_value(
                obs1,
                info1, 
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
            info: dict, 
            info1: dict, 
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
            info, 
            info1, 
            reward, 
            done, 
            env
        )
        if self._debug > 1:
            print(f"\n\nupdate_value {update_value} for obs {obs}, info {info[self._info_field][-1]}")
        self._set_action_state_value(
            obs, 
            info, 
            action, 
            update_value, 
        )

    def __train_one_episode(
            self,
            env: gym.Env
        ) -> tuple[int]:
        obs, info = env.reset()
        if self._debug > 1:
            print(f"obs {obs} info {info}")
        episode_length = 0
        while True:
            action = self.policy(
                obs,
                info, 
                count_action_type = True
            )
            self._increment_counter_action_state(
                obs, 
                info,
                action   
            )
            obs1, reward, done, _, info1 = env.step(action)
            if self._debug:
                print(f"\n\n__train_one_episode\n\tobs {obs} info {info}, done {done}\n\ttype(obs) {type(obs)} type(info) {type(info)} type(done) {type(done)}")
            self.__update_action_state_value(
                action,
                obs, 
                obs1, 
                info, 
                info1, 
                reward, 
                env, 
                done
            )
            obs = obs1; info = info1
            self.current_score = info1[self._score_field]
            episode_length += 1
            if done:
                break
        return episode_length

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

            width ->  {self._width}
            height -> {self._height}
            
        Size of action space :
            
            n_actions-> {self._n_actions}

        Parameters of the agent: 

            alpha (step_size) -> {self.alpha}
            gamma (discount_factor) -> {self.gamma}
            epsilon (exploration_rate) -> {self.epsilon}
            n_steps -> {self.n_steps}
            prior -> {self._prior}

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
            env: gym.Env, 
            n_episodes: int, 
            patience: int = 1e+6,
            dump: bool = True 
        ) -> tuple[list[int]]:
        episodes_lengths = []
        episodes_scores = []
        episodes_with_no_improvement = 0
        if self.verbose:
            self.__print_training_description_message(
                n_episodes, 
                episodes_scores, 
                True
            )
        for episode in tqdm(range(n_episodes)):
            episode_length = self.__train_one_episode(env)
            episodes_lengths.append(episode_length)
            episodes_scores.append(self.current_score)
            if self.max_score < self.current_score:
                if self.verbose:
                    print(f"New record at episode {episode}:\n\n\t New Best Score: {self.current_score} \t Old Best Score: {self.max_score}")
                self.max_score = self.current_score
                episodes_with_no_improvement = 0
            else: 
                episodes_with_no_improvement += 1
            if episodes_with_no_improvement > patience:
                break
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
        return (episodes_lengths, episodes_scores)

    def train_n_episodes(
            self, 
            env: gym.Env, 
            n_episodes: int,
            patience: int = 1e+6
        ) -> None:
        return self.__train_n_episodes(env, n_episodes)