import time
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np


class ReplayBuffer:

    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]


class MaxSarsaAgent:

    def __init__(self,
                 env,
                 gamma=0.99,
                 learning_rate=0.1,
                 replay_batch_size=32):
        self.env = env
        self.q_values = {(state, action): 0.0
                         for state in range(env.observation_space.n)
                         for action in range(env.action_space.n)}
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer = ReplayBuffer()
        self.replay_batch_size = replay_batch_size

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

    def _train_agent(self):
        training_batch = self.buffer.sample(self.replay_batch_size)
        for state, action, reward, new_state in training_batch:
            self._update_q_table(state, action, reward, new_state)

    def play_n_steps(self, steps=10000, train_with_replay=True):
        current_state = self.env.reset()
        total_rewards = []
        total_reward = 0.0
        episode_counter = 1
        for step in range(steps):
            action = self._get_best_action(current_state)
            new_state, reward, done, _ = self.env.step(action)
            self._update_q_table(current_state, action, reward, new_state)
            self.buffer.add(current_state, action, reward, new_state)
            total_reward += reward
            if train_with_replay:
                self._train_agent()
            if done:
                current_state = self.env.reset()
                total_rewards.append(total_reward)
                total_reward = 0
                episode_counter += 1
            else:
                current_state = new_state

        return total_rewards

    def play_episode(self, render=False):
        current_state = self.env.reset()
        total_reward = 0.0
        step_counter = 1
        while True:
            action = self._get_best_action(current_state)
            render_step(env, render)
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            step_counter += 1
            if done:
                render_step(env, render)
                return total_reward
            current_state = new_state


def render_step(env, render):
    if render:
        print('\n' * 25)
        env.render()
        time.sleep(0.5)


def plot_total_reward_convergence(agent, label, color, steps_to_play=15000, replay=True, zorder=0):
    total_rewards = agent.play_n_steps(steps_to_play, replay)
    plot, = plt.plot(np.arange(len(total_rewards)), total_rewards, color=color, zorder=zorder, label=label)
    return plot


if __name__ == "__main__":
    env = gym.make('Taxi-v2')

    agent_with_replay = MaxSarsaAgent(env)
    agent_without_replay = MaxSarsaAgent(env)

    """ With replay agent needs to make less real steps"""
    with_replay_convergence = plot_total_reward_convergence(agent_with_replay, 'With replay', replay=True, color='k',
                                                            zorder=2)
    without_replay_convergence = plot_total_reward_convergence(agent_without_replay, 'Without replay', replay=False,
                                                               color='g')
    plt.legend(handles=[with_replay_convergence, without_replay_convergence])
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.show()

    for _ in range(10):
        agent_with_replay.play_episode(render=True)
