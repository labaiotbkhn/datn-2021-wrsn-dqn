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
import random
UPDATE_EVERY = 10


class DQN:
    def __init__(self, state_size, file_name_model, nb_action=81, action_func=action_function, network=None, epsilon=1, epsilon_decay=0.8, epsilon_min=0.01, batch_size=32, discount_factor=0.99, num_of_episodes=500):
        self.action_list = action_func(nb_action=nb_action)
        self.state = nb_action

        self.charging_time = [0.0 for _ in self.action_list]
        self.reward_max = [0.0 for _ in self.action_list]
        self.learning_rate = 1e-3
        self.memory = deque(maxlen=5000)
        self.priority = deque(maxlen=5000)
        self.epsilon = epsilon  # exploration rate
        self.discount_factor = discount_factor  # discount rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_size = state_size
        self.action_size = len(self.action_list)
        self.file_name_model = file_name_model
        self.input_state = None
        self.batch_size = 64
        self.steps_to_update_target_model = 0
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.reward = np.asarray([0.0 for _ in self.action_list])
        self.reward_max = [0.0 for _ in self.action_list]
        self.q_value = [0.0 for _ in self.action_list]
        self.time_chosing_by_reward = 0

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self._huber_loss, optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from Main model to Target model
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save_weights(filepath=self.file_name_model)

    # update with replay training with piority
    def prioritize(self, state, next_state, action, reward, alpha=0.6):
        q_next = reward + self.discount_factor * \
            np.max(self.target_model.predict(next_state)[0])
        q = self.model.predict(state)[0][action]
        p = (np.abs(q_next-q) + (np.e ** -10)) ** alpha
        return p

    def get_priority_experience_batch(self):
        piorities_before_last = collections.deque(
            itertools.islice(self.priority, 0, len(self.priority)-1))
        p_sum = np.sum(piorities_before_last)
        prob = piorities_before_last / p_sum
        sample_indices = random.choices(
            range(len(prob)), k=self.batch_size, weights=prob)
        importance = (1/prob) * (1/len(piorities_before_last))
        importance = np.array(importance)[sample_indices]
        samples = np.array(self.memory)[sample_indices]
        return samples, importance

    def memorize(self, state, action, reward, next_state):
        # update memory in Experience Replay
        self.memory.append((state, action, reward, next_state))
        self.priority.append(0)
        pass

    def choose_next_state(self, network, state):
        # next_state = np.argmax(self.q_table[self.state])
        if network.mc.energy < 10:
            return len(self.q_value) - 1
        act_values = self.model.predict(state)
        q_value = act_values[0]
        indices = sorted(range(len(q_value)),
                         key=lambda k: q_value[k], reverse=True)

        # five with element have largest qvalue
        max_reward = max(self.reward)
        chossing_indices = []
        for indice in indices[:5]:
            if(self.reward[indice] > 0.6 * max_reward):
                chossing_indices.append(indice)
        if len(chossing_indices) > 0:
            # max_reward = 0
            # index = 0
            # for indice_reward in chossing_indices:
            #     if self.reward[indice_reward] > max_reward:
            #         index = indice_reward
            #         max_reward = self.reward[indice_reward]
            index = random.choices(
                population=chossing_indices, weights=self.q_value[chossing_indices], k=1)[0]
            print("Chosing with Q_value from DQN: ", index)
            return index
        else:
            index = np.argmax(self.reward)
            print("Chosing with Q_value from Reward: ", index)
            return index
        return np.argmax(q_value)

        # if np.random.rand() <= self.epsilon:
        #     print("choosing with random")
        #     return random.randrange(self.action_size)
        # act_values = self.model.predict(state)
        # q_value = act_values[0]
        # weights = self.create_relative_reward_qvalues(q_value)

        # return np.argmax(weights)

        # if np.max(q_value) > (1+epsilon_deep_value) * second_value:

    def experience_replay(self, batch_size):
        # get minibatch memories from 0 => last memory -1
        print("training with experience, with priority")
        batch, importance = self.get_priority_experience_batch()

        for b, i in zip(batch, importance):
            state, action, reward, next_state = b

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
        print("reward for next_action: {}".format(reward))
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
        # if charging_time > 8000:
        #     charging_time *= 0.65

        # training experience replay with
        self.training_replay()
        print("update weights: ", self.steps_to_update_target_model)
        if (self.steps_to_update_target_model-1) % 50 == 0:
            self.save_model()
        print("next state =({}), {}, charging_time: {}).".format(
            self.action_list[self.state], self.state, charging_time))
        return self.action_list[self.state], charging_time
