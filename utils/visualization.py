import json
import os
from typing import List

from matplotlib import pyplot as plt

from local_constants import MODEL_DIR


def plot_running_results(output_path: str, lrs, accs, loss):
    plt.ioff()
    learning_rates = []
    for entry in lrs:
        for key in entry.keys():
            for val in entry.get(key):
                learning_rates.append(val)

    accuracies = []
    accuracies_steps = []
    for entry in accs:
        for key in entry.keys():
            for val in entry.get(key):
                for step in val.keys():
                    accuracies_steps.append(int(step))
                    accuracies.append(val.get(step))

    losses = []
    losses_steps = []
    for entry in loss:
        for key in entry.keys():
            for val in entry.get(key):
                for step in val.keys():
                    losses_steps.append(int(step))
                    losses.append(val.get(step))

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(learning_rates)
    ax1.set(ylabel='learning rate')

    ax2.plot(accuracies_steps, accuracies, 'orange')
    ax2.set(ylabel='accuracy')

    ax3.plot(losses_steps, losses, 'green')
    ax3.set(ylabel='loss', xlabel='step')
    plt.savefig(output_path)
    plt.close(fig=fig)

    # plt.show()


def plot_performance(x: List[int], y: List[float], measure: str):
    plt.title('loss')
    plt.xlabel('step')
    plt.legend(measure)
    plt.plot(x, y)

    plt.show()


if __name__ == '__main__':

    model_name = 'NbAiLab_nb-bert-base-2022-06-24_13-57-20'

    lr_file = model_name + '/learning_rates'
    acc_file = model_name + '/accuracies'
    loss_file = model_name + '/eval_losses'

    lrs = []
    with open(os.path.join('../' + MODEL_DIR, lr_file + '.json'), "rb") as file:
        for entry in file:
            lrs.append(json.loads(entry))

    accs = []
    with open(os.path.join('../' + MODEL_DIR, acc_file + '.json'), "rb") as file:
        for entry in file:
            accs.append(json.loads(entry))

    loss_init = []
    with open(os.path.join('../' + MODEL_DIR, loss_file + '.json'), "rb") as file:
        for entry in file:
            loss_init.append(json.loads(entry))

    learning_rates = []
    for entry in lrs:
        for key in entry.keys():
            for val in entry.get(key):
                learning_rates.append(val)

    accuracies = []
    accuracies_steps = []
    for entry in accs:
        for key in entry.keys():
            for val in entry.get(key):
                for step in val.keys():
                    accuracies_steps.append(int(step))
                    accuracies.append(val.get(step))

    losses = []
    losses_steps = []
    for entry in loss_init:
        for key in entry.keys():
            for val in entry.get(key):
                for step in val.keys():
                    losses_steps.append(int(step))
                    losses.append(val.get(step))

    plot_running_results(output_path='../plots/test.png', lr=learning_rates,
                         acc_step=accuracies_steps, acc=accuracies,
                         loss_step=losses_steps, loss=losses)

    # plot_performance(x=losses_steps, y=losses, measure='loss')
    # plot_performance(x=accuracies_steps, y=accuracies, measure='acc')
