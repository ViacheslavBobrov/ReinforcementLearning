import gym
import numpy as np

from dqn_agent import DQNAgent


class CartPoleTrainer:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.mean_reward_episode_n = 50
        self.target_model_sync_frequency = 20
        self.target_mean_reward = 250

    def _run(self, test_mode=False, render=False):
        episode_reward = 0.0
        obs = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.agent.act(obs, test_mode)
            new_obs, reward, done, _ = self.env.step(action)
            self.agent.remember(obs, action, reward, done, new_obs)
            episode_reward += reward
            if done:
                break
            obs = new_obs
        return episode_reward

    def train(self, min_episodes=10000):
        episode_rewards = []
        for i in range(min_episodes):
            episode_reward = self._run(test_mode=False, render=False)
            episode_rewards.append(episode_reward)

            reward_mean = np.mean(episode_rewards[-self.mean_reward_episode_n:])
            print('Episode: ', i + 1, 'Mean reward last {0} episodes: '.format(self.mean_reward_episode_n), reward_mean)
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
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.n,
                     hidden_layers_number=3,
                     hidden_layers_size=50,
                     memory_size=1500)
    cart_pole_trainer = CartPoleTrainer(env, agent)
    cart_pole_trainer.train()
    cart_pole_trainer.test()
