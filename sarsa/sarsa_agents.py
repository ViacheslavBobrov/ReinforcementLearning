import numpy as np
from cliff_walking_renderer import CliffWalkingRenderer


class MaxSarsaAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 learning_rate=0.1):
        self.env = env
        self.q_values = {(state, action): 0.0
                         for state in range(env.observation_space.n)
                         for action in range(env.action_space.n)}
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.renderer = CliffWalkingRenderer(env, self.q_values)

    def play_n_steps(self, n_steps, render=False):
        current_state = self.env.reset()
        total_rewards = []
        total_reward = 0.0
        episode_counter = 1
        step_counter = 1
        for _ in range(n_steps):
            action = self._get_best_action(current_state)
            self._render_step(render, current_state, action, False, episode_counter, step_counter)
            new_state, reward, done, _ = self.env.step(action)
            self._update_q_table(current_state, action, reward, new_state)
            total_reward += reward
            step_counter += 1
            if done:
                total_rewards.append(total_reward)
                self._render_step(render, new_state, action, True, episode_counter, step_counter)
                episode_counter += 1
                step_counter = 1
                total_reward = 0
                current_state = self.env.reset()
            else:
                current_state = new_state
        return total_rewards

    def _get_best_action(self, state):
        state_q_values = [self.q_values[(state, action)] for action in range(self.env.action_space.n)]
        best_action = np.argmax(state_q_values)
        return best_action

    def _update_q_table(self, current_state, action, reward, new_state):
        next_best_action = self._get_best_action(new_state)
        update_value = reward + self.gamma * self.q_values[(new_state, next_best_action)] - self.q_values[
            (current_state, action)]
        new_value = self.q_values[(current_state, action)] + self.learning_rate * update_value
        self.q_values[(current_state, action)] = new_value

    def _render_step(self, render, current_state, action, done, episode, step_counter, clear_display=True):
        if render:
            self.renderer.draw_environment(current_state, episode, step_counter, action, done, clear_display)

    def play_episode(self, render=False):
        current_state = self.env.reset()
        total_reward = 0.0
        step_counter = 1
        while True:
            action = self._get_best_action(current_state)
            self._render_step(render, current_state, action, False, 1, step_counter)
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            step_counter += 1
            if done:
                self._render_step(render, new_state, action, True, 1, step_counter, clear_display=False)
                return total_reward
            current_state = new_state


class ExpectedSarsaAgent(MaxSarsaAgent):

    def __init__(self,
                 env,
                 gamma=0.99,
                 exploration_decay=0.9997,
                 exploration_min=0.00,
                 learning_rate=0.1):
        super().__init__(env, gamma, learning_rate)
        self.epsilon = 1.0
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def _decay_exploration(self):
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay

    def _get_best_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return super()._get_best_action(state)

    def _update_q_table(self, current_state, action, reward, new_state):
        self._decay_exploration()
        next_best_action = self._get_best_action(new_state)
        update_value = reward + self.gamma * self.q_values[(new_state, next_best_action)] - self.q_values[
            (current_state, action)]
        new_value = self.q_values[(current_state, action)] + self.learning_rate * update_value
        self.q_values[(current_state, action)] = new_value
