import random

import numpy as np
import tensorflow as tf


class ReplayMemory:

    def __init__(self, capacity, transition_length):
        self.size = capacity
        self.memory = np.zeros((capacity, transition_length), dtype=np.float32)
        self.pointer = 0

    def remember(self, state, action, reward, next_state):
        if self.pointer == self.size:
            self.pointer = 0
        transition = np.hstack((state, action, reward, next_state))
        self.memory[self.pointer, :] = transition
        self.pointer += 1

    def sample(self, batch_size=32):
        current_population = self.size if self.pointer > self.size else self.pointer
        indices = np.random.choice(current_population, size=batch_size)
        batch = self.memory[indices, :]
        return batch


class TicTacToeAgent:

    def __init__(self, env, sess, name,
                 learning_rate=0.01,
                 gamma=0.99,
                 batch_size=32,
                 target_net_update_frequency=500,
                 epsilon_exploration=0.05):
        self.env = env
        self.sess = sess
        self.name = name
        self.n_actions = 9
        self.n_observations = 9

        self.memory = ReplayMemory(10000, self.n_observations * 2 + 1 + 1)
        self.batch_size = batch_size
        self.target_net_update_frequency = target_net_update_frequency
        self.learn_step_counter = 0
        self.epsilon_exploration = epsilon_exploration

        self._build_dqn(learning_rate, gamma, name)

    def _build_dqn(self, learning_rate, gamma, name):
        self.state = tf.placeholder(tf.float32, [None, self.n_observations])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.reward = tf.placeholder(tf.float32, [None, ])
        self.state_next = tf.placeholder(tf.float32, [None, self.n_observations])

        with tf.variable_scope(name):
            with tf.variable_scope('dqn_eval'):
                layer1_eval = tf.layers.dense(self.state, 64, activation=tf.nn.relu)
                self.q_values_eval = tf.layers.dense(layer1_eval, self.n_actions, activation=tf.nn.softmax)

            with tf.variable_scope('dqn_target'):
                layer1_target = tf.layers.dense(self.state_next, 64, activation=tf.nn.relu)
                q_values_target = tf.layers.dense(layer1_target, self.n_actions, activation=tf.nn.softmax)

        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + '/dqn_target')
        self.eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + '/dqn_eval')
        self.dqn_target_replace_op = [tf.assign(t, e) for t, e in zip(self.target_params, self.eval_params)]

        q_value_next = self.reward + gamma * tf.reduce_max(q_values_target, axis=1)
        action_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        q_value_predicted = tf.gather_nd(self.q_values_eval, action_indices)

        cost = tf.losses.mean_squared_error(labels=q_value_next, predictions=q_value_predicted)
        self.dqn_training_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    def get_params(self):
        return self.target_params + self.eval_params

    def update(self):
        if self.learn_step_counter % self.target_net_update_frequency == 0:
            self.sess.run(self.dqn_target_replace_op)

        batch = self.memory.sample(self.batch_size)
        batch_states = batch[:, :self.n_observations]
        batch_actions = batch[:, self.n_observations]
        batch_rewards = batch[:, self.n_observations + 1]
        batch_states_next = batch[:, -self.n_observations:]
        self.sess.run(self.dqn_training_op, feed_dict={self.state: batch_states,
                                                       self.action: batch_actions,
                                                       self.reward: batch_rewards,
                                                       self.state_next: batch_states_next})
        self.learn_step_counter += 1

    def act(self, state, epsilon_exploration=0.0):
        free_cells, occupied_cells = self.env.get_free_and_occupied_cells()
        epsilon_exploration = max(self.epsilon_exploration, epsilon_exploration)
        if np.random.random() > epsilon_exploration:
            q_values = self.sess.run(self.q_values_eval, {self.state: state[np.newaxis, :]})[0]
            np.put(q_values, occupied_cells, -1)  # only legal moves
            return np.argmax(q_values)
        else:
            return random.choice(free_cells)  # only legal moves

