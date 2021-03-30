from utils import updateMemories, updateNextAction, _build_input_state
from scipy.spatial import distance

import Parameter as para
from MobileCharger_Method import get_location, charging
import numpy as np
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

    def get_next_location(self, network, time_stem, optimizer=None, deep_optimizer=None):
        # check steps_to_update_target_model in deep Q learning
        # deep_optimizer.steps_to_update_target_model = time_stem

        if(len(deep_optimizer.memory)) > 30:
            print("Update with deep Q leatning")
            next_location, charging_time = deep_optimizer.update(network)
        # collect state for DQN and switch to use DQN if len(memmories) > 30
        else:
            print("Q learning update")
            next_location, charging_time = optimizer.update(network)
            next_state_last_memories_dqn = optimizer.input_state_dqn
            next_state_last_memories_dqn = np.reshape(next_state_last_memories_dqn, [1, deep_optimizer.state_size])
            print("reward for last memories_dqn: ", optimizer.reward_dqn)
            reward_last_memories_dqn = optimizer.reward_dqn
            # update last_memory

            updateNextAction(deep_optimizer, next_state_last_memories_dqn, reward_last_memories_dqn)
            updateMemories(optimizer, deep_optimizer)
            deep_optimizer.training_replay()

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
