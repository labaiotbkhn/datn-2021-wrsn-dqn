import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
import random
from collections import deque
from Q_learning_method import *
from utils import _build_input_state


class DQN:
    def __init__(self, state_size, file_name_model, nb_action=81, action_func=action_function, network=None, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, discount_factor=0.9, num_of_episodes=500):
        self.action_list = action_func(nb_action=nb_action)
        self.state = nb_action

        self.charging_time = [0.0 for _ in self.action_list]
        self.reward = 0
        self.reward_max = [0.0 for _ in self.action_list]
        self.learning_rate = 1e-5
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon  # exploration rate
        self.discount_factor = discount_factor  # discount rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_size = state_size
        self.action_size = len(self.action_list)
        self.file_name_model = file_name_model
        self.input_state = None
        self.batch_size = 8
        self.steps_to_update_target_model = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.reward = np.asarray([0.0 for _ in self.action_list])
        self.reward_max = [0.0 for _ in self.action_list]

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(
            lr=self.learning_rate), metrics=['accuracy'])
        return model

    def update_target_model(self):
        # copy weights from Main model to Target model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state):
        # update memory in Experience Replay
        self.memory.append((state, action, reward, next_state))
        pass

    def choose_next_state(self, network, state):
        # next_state = np.argmax(self.q_table[self.state])
        if network.mc.energy < 10:
            return len(self.q_table) - 1

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def experience_replay(self, batch_size):
        # get minibatch memories from 0 => last memory -1
        minibatch = random.sample(self.memory[:len(self.memory)-1], batch_size)

        for state, action, reward, next_state in minibatch:
            # predict state cordination (X, y) of MC
            target = self.model.predict(state)
            t = self.target_model.predict(next_state)[0]

            target[0][action] = reward + self.discount_factor * np.amax(t)

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def training_replay(self):
        if len(self.memory) >= self.batch_size:
            print("trainin with replay: ", self.steps_to_update_target_model)
            self.experience_replay(self.batch_size)
        pass

    def load(self, name):
        self.model.load_weights(name)

    def set_reward(self, reward_func=reward_function, network=None,):
        # create reward with state
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index in range(len(self.action_list)):
            temp = reward_func(network=network, q_learning=self,
                               state=index, receive_func=find_receiver)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        self.reward = first + second + third
        self.reward_max = list(zip(first, second, third))

    def update(self, network, alpha=0.5, gamma=0.5, q_max_func=q_max_function, reward_func=reward_function):
        if not len(network.mc.list_request):
            return self.action_list[self.state], 0.0
        # if self.input_state is None:
        #     self.input_state  = network.mc.current

        self.input_state = _build_input_state(network)
        self.input_state = np.reshape(self.input_state, [1, self.state_size])
        # update next_state in last memories:
        self.memory[-1][3] = self.input_state
        next_action_id = self.choose_next_state(network, self.input_state)

        # calculate reward
        self.set_reward(reward_func=reward_func, network=network)
        reward = self.reward[next_action_id]

        # update memories temporary
        self.memorize(self.input_state, self.state, reward, self.input_state)
        # update input_state
        # self.input_state = next_state
        self.state = next_action_id

        # calculate charging time
        if self.state == len(self.action_list) - 1:
            charging_time = (network.mc.capacity -
                             network.mc.energy) / network.mc.e_self_charge
        else:
            charging_time = self.charging_time[next_action_id]

        # training experience replay with
        self.training_replay()

        # update weights target after time % 100 == 0 update
        if (self.steps_to_update_target_model - 1) % 100 == 0:
            self.update_target_model()
            print("update weights: ", self.steps_to_update_target_model)
        print("next state =",
              self.action_list[self.state], self.state, charging_time)

        return self.action_list[self.state], charging_time
