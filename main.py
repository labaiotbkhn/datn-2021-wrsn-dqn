from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger
from Q__Learning import Q_learning
from DQN import DQN
from Inma import Inma
import csv
from scipy.stats import sem, t
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filedata", help="File name to run data",
                    type=str)
parser.add_argument("--type", type=str)
parser.add_argument("--maxTime", default=None, type=int)
parser.add_argument("--model", default="DQN", type=str)
args = parser.parse_args()


def calculate_state_size_dqn(network):
    # energy of sensor
    # comsumption rate of sensor
    return len(network.node) * 2


df = pd.read_csv(args.filedata)
for index in range(5):
    chooser_alpha = open("log/{}.csv".format(args.type), "w")
    f = open("result.txt", "w+")
    result = csv.DictWriter(chooser_alpha, fieldnames=["nb run", "lifetime"])
    result.writeheader()
    life_time = []
    for nb_run in range(5):
        random.seed(nb_run)

        node_pos = list(literal_eval(df.node_pos[index]))
        list_node = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            com_ran = df.commRange[index]
            energy = df.energy[index]
            energy_max = df.energy[index]
            prob = df.freq[index]
            node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                        energy_thresh=0.4 * energy, prob=prob)
            list_node.append(node)
        mc = MobileCharger(energy=df.E_mc[index], capacity=df.E_max[index], e_move=df.e_move[index],
                           e_self_charge=df.e_mc[index], velocity=df.velocity[index])
        target = [int(item) for item in df.target[index].split(',')]
        net = Network(list_node=list_node, mc=mc, target=target)

        q_learning = Q_learning(network=net)
        inma = Inma()

        # calculate state_size for DQN
        state_size = calculate_state_size_dqn(net)
        deep_qlearning = DQN(state_size=state_size,
                             file_name_model="model.h5", network=net)
        if args.model == "DQN":
            file_name = "log/{}_{}_{}_{}.csv".format(
                args.model, args.type, index, nb_run)
            temp = net.simulate(optimizer=q_learning,
                                file_name=file_name, deep_optimizer=deep_qlearning, max_time=args.maxTime)
        elif args.model == "Q-learning":
            file_name = "log/{}_{}_{}_{}.csv".format(
                args.model, args.type, index, nb_run)
            temp = net.simulate(optimizer=q_learning,
                                file_name=file_name, deep_optimizer=None, max_time=args.maxTime)
        else:
            file_name = "log/{}_{}_{}_{}.csv".format(
                args.model, args.type, index, nb_run)
            temp = net.simulate(optimizer=inma,
                                file_name=file_name, deep_optimizer=None, max_time=args.maxTime)

        # inma = Inma()
        # q_learning = DQN(state_size=2,file_name_model="model.h5", network=net)

        life_time.append(temp)
        result.writerow({"nb run": nb_run, "lifetime": temp})
        f.write("nb run: {}; lifetime: {}".format(nb_run, temp))

    confidence = 0.95
    h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
    result.writerow({"nb run": np.mean(life_time), "lifetime": h})
    f.write("nb run: {}; lifetime: {}".format(np.mean(life_time), h))
    print(np.mean(life_time))
    print(h)
