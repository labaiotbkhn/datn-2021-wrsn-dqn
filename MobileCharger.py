from utils import updateMemories, updateNextAction, _build_input_state
from scipy.spatial import distance

import Parameter as para
from MobileCharger_Method import get_location, charging
import numpy as np
import random


class MobileCharger:
    def __init__(self, energy=None, e_move=None, start=para.depot, end=para.depot, velocity=None,
                 e_self_charge=None, capacity=None):
        self.is_stand = False  # is true if mc stand and charge
        self.is_self_charge = False  # is true if mc is charged
        self.is_active = False

        self.start = start  # from location
        self.end = end  # to location
        self.current = start  # location now
        self.end_time = -1

        self.energy = energy  # energy now
        self.capacity = capacity  # capacity of mc
        self.e_move = e_move  # energy for moving
        self.e_self_charge = e_self_charge  # energy receive per second
        self.velocity = velocity  # velocity of mc

        self.list_request = []

    def update_location(self, func=get_location):
        self.current = func(self)
        self.energy -= self.e_move

    def charge(self, net=None, node=None, func=charging):
        func(self, net, node)

    def self_charge(self):
        self.energy = min(self.energy + self.e_self_charge, self.capacity)

    def check_state(self):
        if distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        else:
            self.is_stand = False
        if distance.euclidean(para.depot, self.end) < 10 ** -3:
            self.is_self_charge = True
        else:
            self.is_self_charge = False

    def choice_optimizer(self, network, index_optimizer, optimizer=None, deep_optimizer=None,):
        if index_optimizer == 1:
            print("Q learning update", deep_optimizer.steps_to_update_target_model)
            next_location, charging_time = optimizer.update(network)
            
            next_state_last_memories_dqn = optimizer.input_state_dqn
            next_state_last_memories_dqn = np.reshape(next_state_last_memories_dqn, [
                                                      1, deep_optimizer.state_size])
            # update last_memory
            updateNextAction(
                deep_optimizer, next_state_last_memories_dqn)
            updateMemories(optimizer, deep_optimizer)

            deep_optimizer.updateWeightFromQLearning(
                next_state_last_memories_dqn, optimizer.q_value_for_dqn)
            
            return next_location, charging_time
        else:
            print("Update with DQN", deep_optimizer.steps_to_update_target_model)
            next_location, charging_time = deep_optimizer.update(network)
            return next_location, charging_time

    def get_next_location(self, network, time_stem, optimizer=None, deep_optimizer=None):
        list_optimizer = [1, 2]  # 1 - Qlearning; 2-DeepLearning
        index_optimizer = 1
        if deep_optimizer.steps_to_update_target_model < 300:
            index_optimizer = 1
        elif deep_optimizer.steps_to_update_target_model >= 300 and deep_optimizer.steps_to_update_target_model < 500:
            index_optimizer = random.choices(
                list_optimizer, weights=(75, 25), k=1)[0]
        elif deep_optimizer.steps_to_update_target_model >= 500 and deep_optimizer.steps_to_update_target_model < 650:
            index_optimizer = random.choices(
                list_optimizer, weights=(50, 50), k=1)[0]
        else:
            index_optimizer = 2
        next_location, charging_time = self.choice_optimizer(
            network, index_optimizer, optimizer, deep_optimizer)
        self.start = self.current
        self.end = next_location
        moving_time = distance.euclidean(self.start, self.end) / self.velocity
        self.end_time = time_stem + moving_time + charging_time

    def run(self, network, time_stem, net=None, optimizer=None, deep_optimizer=None):
        # print(self.energy, self.start, self.end, self.current)
        if (not self.is_active and self.list_request) or abs(
                time_stem - self.end_time) < 1:
            self.is_active = True
            self.list_request = [request for request in self.list_request if
                                 net.node[request["id"]].energy < net.node[request["id"]].energy_thresh]
            if not self.list_request:
                self.is_active = False

            self.get_next_location(network=network, time_stem=time_stem,
                                   optimizer=optimizer, deep_optimizer=deep_optimizer)
        else:
            if self.is_active:
                if not self.is_stand:
                    # print("moving")
                    self.update_location()
                elif not self.is_self_charge:
                    # print("charging")
                    self.charge(net)
                else:
                    # print("self charging")
                    self.self_charge()
        if self.energy < para.E_mc_thresh and not self.is_self_charge and self.end != para.depot:
            self.start = self.current
            self.end = para.depot
            self.is_stand = False
            charging_time = self.capacity / self.e_self_charge
            moving_time = distance.euclidean(
                self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()
