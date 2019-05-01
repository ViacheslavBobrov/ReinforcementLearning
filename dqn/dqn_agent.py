import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:

    def __init__(self, state_size,
                 action_size,
                 memory_size,
                 hidden_layers_number,
                 hidden_layers_size,
                 learning_rate=0.001,
                 gamma=0.95,
                 sample_batch_size=32,
                 exploration_rate=1.0,
                 exploration_min=0.01,
                 exploration_decay=0.995):
        assert hidden_layers_number > 0

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.sample_batch_size = sample_batch_size
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.model = self._build_model(hidden_layers_number, hidden_layers_size)
        self.target_model = self._build_model(hidden_layers_number, hidden_layers_size)

    def _build_model(self, hidden_layers_number, hidden_layers_size):
        model = Sequential()
        model.add(Dense(hidden_layers_size, activation='relu', input_dim=self.state_size))
        for i in range(hidden_layers_number - 1):
            model.add(Dense(hidden_layers_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    def sync_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self):
        """ Double DQN """
        if len(self.memory) < self.sample_batch_size:
            return

        batch = random.sample(self.memory, self.sample_batch_size)
        states, actions, rewards, dones, next_states = unpack_batch(batch)

        next_state_values_model_indexes = np.argmax(self.target_model.predict(next_states), axis=1)
        next_state_values_target_model = self.target_model.predict(next_states)

        next_state_values = np.zeros(len(states))
        for i, index in enumerate(next_state_values_model_indexes):
            next_state_values[i] = next_state_values_target_model[i, index]

        # setting values to 0 for episodes that are done. Only rewards should be taken into calculation in this case
        next_state_values *= 1 - dones
        targets = next_state_values * self.gamma + rewards
        # To calculate MSE based only on target (maximum) action values for each state, let's make MSE for the rest
        # action values to be equal 0. For this lets predict all action values for states and replace those that are
        # expected to be target(maximum) with values calculated by Bellman's equation
        expected_state_action_values = self.model.predict(states)
        for i in range(len(expected_state_action_values)):
            expected_state_action_values[i, actions[i]] = targets[i]

        self.model.fit(states, expected_state_action_values, epochs=1, verbose=0, batch_size=1)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def act(self, state, test_mode=False):
        if (np.random.rand() <= self.exploration_rate) & (not test_mode):
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array(state).reshape((1, self.state_size)))
        return np.argmax(act_values[0])


def unpack_batch(batch):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for state, action, reward, done, next_state in batch:
        state = np.array(state, copy=False)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if next_state is None:
            next_states.append(state)  # the result will be masked anyway
        else:
            next_states.append(np.array(next_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(next_states, copy=False)
