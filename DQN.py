import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
import random
from collections import deque
from Q_learning_method import *
from utils import _build_input_state, updateNextAction
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras import backend as K

UPDATE_EVERY = 20


class DQN:
    def __init__(self, state_size, file_name_model, nb_action=81, action_func=action_function, network=None, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, discount_factor=0.99, num_of_episodes=500):
        self.action_list = action_func(nb_action=nb_action)
        self.state = nb_action

        self.charging_time = [0.0 for _ in self.action_list]
        self.reward_max = [0.0 for _ in self.action_list]
        self.learning_rate = 1e-3
        self.memory = deque(maxlen=2000)
        self.epsilon = epsilon  # exploration rate
        self.discount_factor = discount_factor  # discount rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_size = state_size
        self.action_size = len(self.action_list)
        self.file_name_model = file_name_model
        self.input_state = None
        self.batch_size = 32
        self.steps_to_update_target_model = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.reward = np.asarray([0.0 for _ in self.action_list])
        self.reward_max = [0.0 for _ in self.action_list]
        self.q_value = [0.0 for _ in self.action_list]

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self._huber_loss, optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from Main model to Target model
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save_weights(filepath=self.file_name_model)

    def memorize(self, state, action, reward, next_state):
        # update memory in Experience Replay
        self.memory.append((state, action, reward, next_state))
        pass

    def choose_next_state(self, network, state):
        # next_state = np.argmax(self.q_table[self.state])
        if network.mc.energy < 10:
            return len(self.q_table) - 1

        # if np.random.rand() <= self.epsilon:
        #     print("choosing with random")
        #     return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        self.q_value = act_values[0]
        print("Q-value from deep Q learning: ", self.q_value)

        # update action with custom policy
        epsilon_deep_value = 0.5
        if np.max(self.q_value) > (1+epsilon_deep_value) * abs(np.mean(self.q_value)):
            print("Chosing with DQN value: ", np.argmax(act_values[0]))
            return np.argmax(act_values[0])
        print("Chosing with Reward value: ", self.reward)
        return np.argmax(self.reward)

    def experience_replay(self, batch_size):
        # get minibatch memories from 0 => last memory -1
        minibatch = random.sample(collections.deque(
            itertools.islice(self.memory, 0, len(self.memory)-1)), batch_size)

        for state, action, reward, next_state in minibatch:

            target = self.model.predict(state)
            t = self.target_model.predict(next_state)[0]

            target[0][action] = reward + self.discount_factor * np.amax(t)

            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def training_replay(self):
        if len(self.memory) > self.batch_size:
            self.steps_to_update_target_model += 1
            print("trainin with replay: ", self.steps_to_update_target_model)
            self.experience_replay(self.batch_size)
            if (self.steps_to_update_target_model - 1) % UPDATE_EVERY == 0:
                self.update_target_model()
        pass

    def load(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.target_model.save_weights(name)

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

    def updateWeightFromQLearning(self, state, qValue):
        print("update weights deep NN from q-learning",
              self.steps_to_update_target_model)
        self.steps_to_update_target_model += 1
        qValue = np.reshape(qValue, [1, len(qValue)])
        self.model.fit(state, qValue, epochs=1, verbose=0)
        if(self.steps_to_update_target_model % UPDATE_EVERY == 0):
            self.update_target_model()

    def update(self, network, alpha=0.5, gamma=0.5, q_max_func=q_max_function, reward_func=reward_function):
        if not len(network.mc.list_request):
            return self.action_list[self.state], 0.0

        self.input_state = _build_input_state(network)
        self.input_state = np.reshape(self.input_state, [1, self.state_size])

        # calculate reward for next_action in  current_state
        self.set_reward(reward_func=reward_func, network=network)

        # print("all reward deep_q_learning of current_state")
        # print(self.reward_max)
        # update next_state + reward in last memories:
        updateNextAction(self, self.input_state)

        next_action_id = self.choose_next_state(network, self.input_state)
        reward = self.reward[next_action_id]
        # update memories temporary
        self.memorize(self.input_state, next_action_id,
                      reward, self.input_state)
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
        print("update weights: ", self.steps_to_update_target_model)
        if (self.steps_to_update_target_model-1) % 50 == 0:
            self.save_model()
        print("next state =({}), {}, charging_time: {}).".format(
            self.action_list[self.state], self.state, charging_time))
        return self.action_list[self.state], charging_time
