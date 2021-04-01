import csv

from scipy.spatial import distance

import Parameter as para
from Network_Method import uniform_com_func, to_string, count_package_function


class Network:
    def __init__(self, list_node=None, mc=None, target=None):
        self.node = list_node
        self.set_neighbor()
        self.set_level()
        self.mc = mc
        self.target = target

    def set_neighbor(self):
        for node in self.node:
            for other in self.node:
                if other.id != node.id and distance.euclidean(node.location, other.location) <= node.com_ran:
                    node.neighbor.append(other.id)

    def set_level(self):
        queue = []
        for node in self.node:
            if distance.euclidean(node.location, para.base) < node.com_ran:
                node.level = 1
                queue.append(node.id)
        while queue:
            for neighbor_id in self.node[queue[0]].neighbor:
                if not self.node[neighbor_id].level:
                    self.node[neighbor_id].level = self.node[queue[0]].level + 1
                    queue.append(neighbor_id)
            queue.pop(0)

    def communicate(self, func=uniform_com_func):
        return func(self)

    def run_per_second(self, t, optimizer, deep_optimizer):
        state = self.communicate()
        request_id = []
        for index, node in enumerate(self.node):
            if node.energy < node.energy_thresh:
                node.request(mc=self.mc, t=t)
                request_id.append(index)
            else:
                node.is_request = False
        if request_id:
            for index, node in enumerate(self.node):
                if index not in request_id and (t - node.check_point[-1]["time"]) > 50:
                    node.set_check_point(t)

        # if len(deep_optimizer.memory) > 30:

        if optimizer:
            self.mc.run(network=self, time_stem=t, net=self,
                        optimizer=optimizer,  deep_optimizer=deep_optimizer)
        return state

    def simulate_lifetime(self, optimizer, deep_optimizer, file_name="log/energy_log.csv"):
        energy_log = open(file_name, "w")
        writer = csv.DictWriter(energy_log, fieldnames=[
                                "time", "mc energy", "min energy"])
        writer.writeheader()
        t = 0
        while self.node[self.find_min_node()].energy >= 0:
            t = t + 1
            if (t-1) % 100 == 0:
                print(t, self.mc.current,
                      self.node[self.find_min_node()].energy)
                self.check_top_number_nodes(5)
            state = self.run_per_second(t, optimizer, deep_optimizer)
            if not (t - 1) % 50:
                writer.writerow(
                    {"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
        print(t, self.mc.current, self.node[self.find_min_node()].energy)
        writer.writerow({"time": t, "mc energy": self.mc.energy,
                         "min energy": self.node[self.find_min_node()].energy})
        energy_log.close()
        return t

    def simulate_max_time(self, optimizer, deep_optimizer, max_time=10000, file_name="log/information_log.csv"):
        information_log = open(file_name, "w")
        writer = csv.DictWriter(information_log, fieldnames=[
                                "time", "nb dead", "nb package"])
        writer.writeheader()
        nb_dead = 0
        nb_package = len(self.target)
        t = 0
        while t <= max_time:
            t += 1
            # optimizer.steps_to_update_target_model = t
            if (t-1) % 100 == 0:
                print(t, self.mc.current,
                      self.node[self.find_min_node()].energy)
                self.check_top_number_nodes(5)
            state = self.run_per_second(t, optimizer, deep_optimizer)
            current_dead = self.count_dead_node()
            current_package = self.count_package()
            if current_dead != nb_dead or current_package != nb_package:
                nb_dead = current_dead
                nb_package = current_package
                writer.writerow(
                    {"time": t, "nb dead": nb_dead, "nb package": nb_package})
                print("time: {}, nb dead: {}, nb package:{}".format(
                    t, nb_dead, nb_package))
        print(t, self.mc.current, self.node[self.find_min_node()].energy)
        information_log.close()
        return t

    def simulate(self, optimizer, deep_optimizer, max_time=None, file_name="log/energy_log.csv"):
        if max_time:
            life_time = self.simulate_max_time(
                optimizer=optimizer, deep_optimizer=deep_optimizer, max_time=max_time, file_name=file_name)
        else:
            life_time = self.simulate_lifetime(
                optimizer=optimizer, deep_optimizer=deep_optimizer, file_name=file_name)
        return life_time

    def print_net(self, func=to_string):
        func(self)

    def find_min_node(self):
        min_energy = 10 ** 10
        min_id = -1
        for node in self.node:
            if node.energy < min_energy:
                min_energy = node.energy
                min_id = node.id
        return min_id

    def count_dead_node(self):
        count = 0
        for node in self.node:
            if node.energy < 0:
                count += 1
        return count

    def count_package(self, count_func=count_package_function):
        count = count_func(self)
        return count

    def check_top_number_nodes(self, number=5):
        distances = [distance.euclidean(
            node.location, self.mc.current) for node in self.node]
        indexs = [i[0]
                  for i in sorted(enumerate(distances), key=lambda x:x[1])][:number]
        print("top 10 node nearest MC")
        for i in indexs:
            print("The node {} with energy is {}. the MC location is {}".format(
                self.node[i].location, self.node[i].energy,  self.mc.current))
