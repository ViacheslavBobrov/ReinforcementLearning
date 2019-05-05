import os

import numpy as np
import tensorflow as tf

from tic_tac_toe.agent import TicTacToeAgent
from tic_tac_toe.environment import EpisodeStateCode, TickTacToeEnvironment


class AgentData:

    def __init__(self, id, model, n_wins=0, last_action=None, last_observation=None):
        self.id = id
        self.model = model
        self.last_action = last_action
        self.last_observation = last_observation
        self.n_wins = n_wins


class TicTacToeAgentTrainer:

    def __init__(self, sess, env, agent1_model, agent2_model, weights_folder_path,
                 win_reward=5,
                 lose_reward=-5,
                 draw_reward=-1,
                 step_reward=0,
                 epsilon_exploration=1.0,
                 epsilon_minimum=0.05,
                 epsilon_decay=0.000001):
        self.sess = sess
        self.env = env
        self.agent1_data = AgentData(id=1, model=agent1_model)
        self.agent2_data = AgentData(id=2, model=agent2_model)

        self.unique_games = []
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.step_reward = step_reward

        self._create_agent_sync_op(agent1_model, agent2_model)
        self.weights_saver = tf.train.Saver(save_relative_paths=True)
        self.weights_path = os.path.join(weights_folder_path, 'weights', 'tic_tac_toe.ckpt')
        self.save_frequency = 20

        self.epsilon_exploration = epsilon_exploration
        self.epsilon_minimum = epsilon_minimum
        self.epsilon_decay = epsilon_decay

    def _create_agent_sync_op(self, agent1, agent2):
        self.agent1_to_2_sync = [tf.assign(a2, a1) for a2, a1 in zip(agent2.get_params(), agent1.get_params())]
        self.agent2_to_1_sync = [tf.assign(a1, a2) for a1, a2 in zip(agent1.get_params(), agent2.get_params())]

    def _sync_agents(self, episode, agent1, agent2):
        if episode % 100 == 0:
            if agent1.n_wins > agent2.n_wins:
                self.sess.run(self.agent1_to_2_sync)
            else:
                self.sess.run(self.agent2_to_1_sync)

    def _render(self):
        n_unique_games = len(self.unique_games)
        if n_unique_games > 5000 and n_unique_games % 1000 == 0:
            self.env.render()

    def _print_statistics_and_save_models(self, episode, episode_state_code):
        unique_game = True
        for game in self.unique_games:
            if np.array_equal(game, self.env.game_cells):
                unique_game = False
                break
        if unique_game:
            self.unique_games.append(self.env.game_cells)
            print('Ep:', episode,
                  '\tAgent1 wins:', self.agent1_data.n_wins,
                  '\tAgent2 wins:', self.agent2_data.n_wins,
                  '\tOutcome:', EpisodeStateCode(episode_state_code).name,
                  '\tUnique games:', len(self.unique_games),
                  '\te:', self.epsilon_exploration)
            if len(self.unique_games) % self.save_frequency == 0:
                self.weights_saver.save(self.sess, self.weights_path)
                print('Models saved')

    def _store_win_memory(self, current_agent, competitor_agent, current_state, next_state, episode_state_code):
        if episode_state_code == EpisodeStateCode.WIN:
            current_agent.n_wins += 1
            current_agent_reward, competitor_agent_reward = self.win_reward, self.lose_reward
        else:
            current_agent_reward, competitor_agent_reward = self.draw_reward, self.draw_reward

        current_agent.model.memory.remember(current_state,
                                            current_agent.last_action,
                                            current_agent_reward, next_state)
        competitor_agent.model.memory.remember(competitor_agent.last_observation,
                                               competitor_agent.last_action,
                                               competitor_agent_reward, next_state)

    def train(self, n_episodes, sync_agents=False, render=False):

        for episode in range(n_episodes):
            self.agent1_data.last_action = None
            self.agent2_data.last_action = None
            self.agent1_data.last_observation, self.agent2_data.last_observation = None, None
            agents = [self.agent1_data, self.agent2_data] if episode % 2 == 0 else [self.agent2_data,
                                                                                    self.agent1_data]

            step_counter = 1
            current_state = self.env.reset()
            while True:
                current_agent, competitor_agent = agents[0], agents[1]

                if render:
                    self._render()
                current_agent.last_action = current_agent.model.act(current_state, self.epsilon_exploration)
                next_state, episode_state_code = self.env.step(current_agent.last_action, current_agent.id,
                                                               competitor_agent.id)

                if episode_state_code != EpisodeStateCode.IN_PROGRESS:
                    if render:
                        self._render()

                    if sync_agents:
                        self._sync_agents(episode, current_agent, competitor_agent)

                    self._store_win_memory(current_agent, competitor_agent,
                                           current_state, next_state, episode_state_code)

                    self._print_statistics_and_save_models(episode, episode_state_code)

                if episode > 1:
                    current_agent.model.update()
                    competitor_agent.model.update()

                if self.epsilon_exploration > self.epsilon_minimum:
                    self.epsilon_exploration -= self.epsilon_decay

                if episode_state_code != EpisodeStateCode.IN_PROGRESS:
                    break

                if step_counter > 2:
                    competitor_agent.model.memory.remember(competitor_agent.last_observation,
                                                           competitor_agent.last_action,
                                                           self.step_reward, next_state)

                current_agent.last_observation = current_state
                current_state = next_state
                agents.reverse()
                step_counter += 1


def dqn_agents_train():
    env = TickTacToeEnvironment()
    with tf.Session() as sess:
        agent1 = TicTacToeAgent(env, sess, 'agent1')
        agent2 = TicTacToeAgent(env, sess, 'agent2')
        sess.run(tf.global_variables_initializer())

        weights_folder_path = os.path.dirname(os.path.abspath(__file__))
        agent_trainer = TicTacToeAgentTrainer(sess, env, agent1, agent2, weights_folder_path)
        agent_trainer.train(1000000, render=True)


if __name__ == "__main__":
    dqn_agents_train()
