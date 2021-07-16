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

        for line in f.readlines():
            k = []
            line = line.split()
            for j in line:
                k.append(float(j))
            output.append(k)
    return np.asarray(output)


def line_chart(models, data_matrix, x_label, y_label, title, xpoints, higher_models=[], name=None, maxx=1.2):
    styles = ['o', 's', 'd', '^', 'x', '*']
    #styles = ['o', 's', 'v', '*']
    line_styles = ['-', '--', '-', '-.', ':', '-', '--']
    # styles = ['o-', '>--', 's-', 'd-.', '^:', 'x-', '*-', 'v-']

    colors = ['#003300', '#009933', '#33cc33',
              '#66ff66', '#99ff99', '#ffffff', '#0033ff']
    barwith = 0.1
    ax1 = plt.subplot(111)

    num_models = data_matrix.shape[0]
    num_x_levels = data_matrix.shape[1]\

    assert num_models == len(
        models), "Number of model must equal to data matrix shape 0"

    lns1 = []
    color_models = {
        "Inma": "b",
        "Q-learning": "g",
        "Deep Q-charging": "r",
    }
    for i, model in enumerate(models):
        if model not in higher_models:
            line = data_matrix[i] / 1000
            print("line", line)
            x = np.arange(num_x_levels)
            fillstyle = 'none'
            if i > 3:
                fillstyle = 'full'
            lni, = ax1.plot(x, line, marker=styles[i % len(
                styles)], markersize=8, color=color_models[model], label=models[i], markevery=1, fillstyle=fillstyle)
            lns1.append(lni)

    ax1.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax1.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax2 = None
    lns2 = []
    count = 0

    for i, model in enumerate(models):
        if model in higher_models:
            line = data_matrix[i]
            x = np.arange(num_x_levels)
            if ax2 is None:
                ax2 = ax1.twinx()
            ln2 = ax2.bar(x + count * barwith, line, width=barwith,
                          color=colors[count],  edgecolor='k', label=model, alpha=0.5)
            lns2.append(ln2)
            count += 1

    plt.xticks(np.arange(len(xpoints)), xpoints, fontsize=10)
    plt.yticks(np.arange(0, 600, step=100), fontsize=10)
    ax1.set_xlim(-0.3, len(xpoints) + .3 - 1)
    ax1.set_ylim(0, maxx + .32)
    # ypoints = np.arange(0, 1.1, 0.2)
    # plt.yticks(np.arange(len(ypoints)), ypoints, fontsize = 16)

    if ax2 is not None:
        ax2.set_xlim(-0.5, len(xpoints) + .5 - 1)
        ax2.set_ylim(0, 0.7)
        ax2.set_yticks(np.arange(0, 0.6, 0.1))
        ax2.tick_params('y', colors='green')

    ax1.grid(True)
    box = ax1.get_position()
    ax1.set_position([box.x0 + 0.02, box.y0 + 0.04, box.width, box.height])

    # plt.legend(ncol = 4,borderaxespad = 0.3, fontsize=10.7)
    plt.legend(ncol=3, fontsize=10, loc=1, columnspacing=2.3)
    plt.savefig(name)
    plt.close()


if __name__ == "__main__":
    # models = ['COMBINE', 'GAlign', 'PALE', 'FINAL', 'REGAL', 'Isorank']
    models = ["Inma", "Q-learning", "Deep Q-charging"]
    mode = "train-ratio"
    datasets = ['thaydoitile']
    xpoint_del_nodes = {"xp": ["0.3", "0.4", "0.5", "0.6", "0.7"],
                        "xlabel": "Ti le gui tin"}
    xpoints = {'train-ratio': xpoint_del_nodes}
    ylabel = "Network lifetime (1000s)"
    maxx = 600
    for dataset in datasets:
        file_name = "data_log/{}.txt".format(dataset)
        name = mode + '-' + dataset
        accs = get_acc(file_name)
        print(accs)
        # import pdb
        # pdb.set_trace()
        #accs = accs.reshape((len(models), -1))
        #accs = np.delete(accs, 1, axis=0)
        xticks = xpoints[mode]["xp"]
        xlabel = xpoints[mode]["xlabel"]
        line_chart(models=models, data_matrix=accs,
                   x_label=xlabel, y_label=ylabel, title=name, xpoints=xticks, higher_models=[], name='figures/{}.png'.format(name), maxx=maxx)

        #line_chart_old(data=accs, xpoints=xticks, xtitle=xlabel, ytitle=ylabel, filename='{}.png'.format(name), models=models, yticks=None, add_Legend=True)
