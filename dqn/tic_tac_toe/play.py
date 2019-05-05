import os
import random

import tensorflow as tf

from tic_tac_toe.agent import TicTacToeAgent
from tic_tac_toe.environment import TickTacToeEnvironment, EpisodeStateCode


class TicTacToeGame:

    def __init__(self, weights_folder):
        self.env = TickTacToeEnvironment()
        self.sess = tf.Session()
        agent1 = TicTacToeAgent(self.env, self.sess, 'agent1')
        agent2 = TicTacToeAgent(self.env, self.sess, 'agent2')
        self.agents = [agent1, agent2]
        reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
        self.saver = tf.train.Saver(reuse_vars_dict)
        self.sess.run(tf.global_variables_initializer())
        weights_path = os.path.join(weights_folder, 'weights', 'tic_tac_toe.ckpt')
        self.saver.restore(self.sess, weights_path)

    def play(self, n_episodes):
        human_first_turn = True
        for _ in range(n_episodes):
            agent = random.choice(self.agents)
            state = self.env.reset()
            episode_state_code = EpisodeStateCode.IN_PROGRESS
            human_move = human_first_turn

            while episode_state_code == EpisodeStateCode.IN_PROGRESS:
                self.env.render()
                if human_move:
                    action = int(input('Please enter move [1-9]:')) - 1
                    state, episode_state_code = self.env.step(action, 1, 2)
                else:
                    action = agent.act(state)
                    _, episode_state_code = self.env.step(action, 2, 2)
                if episode_state_code != EpisodeStateCode.IN_PROGRESS:
                    self.env.render()
                    input('Please press enter to continue')

                human_move = not human_move
            human_first_turn = not human_first_turn


if __name__ == "__main__":
    weights_folder_path = os.path.dirname(os.path.abspath(__file__))
    game = TicTacToeGame(weights_folder_path)
    game.play(10)
