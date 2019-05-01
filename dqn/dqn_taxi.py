import time

import gym
import numpy as np

from dqn_agent import DQNAgent


class TaxiTrainer:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.mean_reward_episode_n = 50
        self.target_model_sync_frequency = 10
        self.target_mean_reward = 1

    def _render(self):
        print('\n' * 25)
        self.env.render()
        time.sleep(0.7)

    def _run(self, test_mode=False, render=False):
        episode_reward = 0.0
        obs = self.env.reset()
        while True:
            if render:
                self._render()
            action = self.agent.act(obs, test_mode)
            new_obs, reward, done, _ = self.env.step(action)
            self.agent.remember(obs, action, reward, done, new_obs)
            episode_reward += reward
            if done:
                if render:
                    self._render()
                break
            obs = new_obs
        return episode_reward

    def train(self, min_episodes=1000):
        episode_rewards = []
        for i in range(min_episodes):
            episode_reward = self._run(test_mode=False, render=False)
            episode_rewards.append(episode_reward)

            reward_mean = np.mean(episode_rewards[-self.mean_reward_episode_n:])
            if i % 10 == 0:
                print('Episode: ', i + 1, 'Reward: ', episode_reward,
                      'Mean reward in last {0} episodes'.format(self.mean_reward_episode_n), reward_mean,
                      'Exploration rate:', self.agent.exploration_rate)
            if reward_mean > self.target_mean_reward:
                break

            self.agent.train()

            if len(episode_rewards) % self.target_model_sync_frequency == 0:
                self.agent.sync_weights()

    def test(self, episodes=100):
        for i in range(episodes):
            episode_reward = self._run(test_mode=True, render=True)
            print('Episode: ', i + 1, 'Reward: ', episode_reward)


if __name__ == "__main__":
    env = gym.make('Taxi-v2')
    agent = DQNAgent(state_size=1,
                     action_size=env.action_space.n,
                     hidden_layers_number=3,
                     hidden_layers_size=50,
                     exploration_decay=0.9998,
                     memory_size=10000)
    taxi_trainer = TaxiTrainer(env, agent)
    taxi_trainer.train(10000)
    taxi_trainer.test()
