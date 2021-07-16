import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import *
import json
import os
import pdb
import re


def get_acc(file_path):
    output = []
    with open(file_path, 'r') as f:
        x_time = []
        y_node_dead = []
        z_current_target = []
        for line in f.readlines():

            line = line.split(",")
            x_time.append(int(line[0]))
            y_node_dead.append(int(line[1]))
            z_current_target.append(int(line[2]))
        return x_time, y_node_dead, z_current_target


def line_chart(models, data_matrix, x_label, y_label, title, higher_models=[], name=None, maxx=1.2):
    styles = ['o', 's', 'd', '^', 'x', '*']
    #styles = ['o', 's', 'v', '*']
    line_styles = ['-', '--', '-', '-.', ':', '-', '--']
    # styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']

    colors = ['#003300', '#009933', '#33cc33',
              '#66ff66', '#99ff99', '#ffffff', '#0033ff']
    barwith = 0.1
    ax1 = plt.subplot(111)

    lns1 = []
    for i, model in enumerate(models):
        if model not in higher_models:
            line = data_matrix[model]["y"]
            print("line", line)
            x = data_matrix[model]["x"]
            fillstyle = 'none'
            if i > 3:
                fillstyle = 'full'
            lni, = ax1.plot(x, line, marker=styles[i % len(
                styles)], markersize=8, color='k', label=models[i], markevery=1, fillstyle=fillstyle)
            lns1.append(lni)

    ax1.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax1.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax2 = None
    lns2 = []
    count = 0

    for i, model in enumerate(models):
        if model in higher_models:
            line = data_matrix[model]["y"]
            x = data_matrix[model]["x"]
            if ax2 is None:
                ax2 = ax1.twinx()
            ln2 = ax2.bar(x + count * barwith, line, width=barwith,
                          color=colors[count],  edgecolor='k', label=model, alpha=0.5)
            lns2.append(ln2)
            count += 1

    plt.xticks(np.arange(0, 90000, step=15000), fontsize=10)
    plt.yticks(np.arange(0, 80, step=10), fontsize=10)
    # ax1.set_xlim(-0.3, len(xpoints) + .3 - 1)
    # ax1.set_ylim(0, maxx + .32)
    # ypoints = np.arange(0, 1.1, 0.2)
    # plt.yticks(np.arange(len(ypoints)), ypoints, fontsize = 16)

    # if ax2 is not None:
    #     ax2.set_xlim(-0.5, len(xpoints) + .5 - 1)
    #     ax2.set_ylim(0, 0.7)
    #     ax2.set_yticks(np.arange(0, 0.6, 0.1))
    #     ax2.tick_params('y', colors='green')

    ax1.grid(True)
    box = ax1.get_position()
    ax1.set_position([box.x0 + 0.02, box.y0 + 0.04, box.width, box.height])

    # plt.legend(ncol = 4,borderaxespad = 0.3, fontsize=10.7)
    plt.legend(ncol=3, fontsize=10, loc=1, columnspacing=2.3)
    plt.savefig(name)
    plt.close()


# models = ["Inma", "Q-learning", "Deep Q-charging"]
# data_matrix = {}
# for model in models:
#     file_name = "data_log/{}_monitor.txt".format(model)
#     name = model + '_monitor_node'
#     x_time, y_nodes, z_target = get_acc(file_name)
#     x_prefix = [i for i in range(x_time[0])]
#     y_prefix = [200 for i in range(x_time[0])]
#     x_time = x_prefix + x_time
#     y_nodes = y_prefix + z_target
#     plt.plot(x_time, y_nodes, label=model)
# plt.xticks(np.arange(0, 90000, step=15000), fontsize=10)
# plt.yticks(np.arange(200, 0, step=-20), fontsize=10)
# plt.xlabel("Time", fontsize=13, fontweight='bold')
# plt.ylabel("current monitored target", fontsize=13, fontweight='bold')
# plt.legend()
# plt.savefig("figures/monitored_target.png")

models = ["Inma", "Q-learning", "Deep Q-charging"]
data_matrix = {}
for model in models:
    file_name = "data_log/{}_monitor.txt".format(model)
    name = model + '_monitor_node'
    x_time, y_nodes, z_target = get_acc(file_name)
    x_prefix = [i for i in range(x_time[0])]
    y_prefix = [0 for i in range(x_time[0])]
    x_time = x_prefix + x_time
    y_nodes = y_prefix + y_nodes
    plt.plot(x_time, y_nodes, label=model)
plt.xticks(np.arange(0, 90000, step=15000), fontsize=10)
plt.yticks(np.arange(0, 80, step=10), fontsize=10)
plt.xlabel("Time", fontsize=13, fontweight='bold')
plt.ylabel("current dead node", fontsize=13, fontweight='bold')
plt.legend()
plt.savefig("figures/monitored_node.png")
