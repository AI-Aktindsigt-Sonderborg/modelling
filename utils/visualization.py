import json
import os
from local_constants import MODEL_DIR
from matplotlib import pyplot as plt
from typing import List

def plot_learning_rates(lr: List[float], acc_step, acc, loss_step, loss):
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(lr)
    ax1.set(ylabel='learning rate')

    ax2.plot(acc_step, acc, 'orange')
    ax2.set(ylabel='accuracy')

    ax3.plot(loss_step, loss, 'green')
    ax3.set(ylabel='loss', xlabel='step')

    plt.show()

def plot_performance(x: List[int], y: List[float], measure: str):
    plt.title('loss')
    plt.xlabel('step')
    plt.legend(measure)
    plt.plot(x, y)

    plt.show()

if __name__ == '__main__':

    model_name = 'NbAiLab_nb-bert-base-2022-06-20_12-38-10'

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


    plot_learning_rates(learning_rates, acc_step=accuracies_steps, acc=accuracies,
                        loss_step=losses_steps, loss=losses)

    # plot_performance(x=losses_steps, y=losses, measure='loss')
    plot_performance(x=accuracies_steps, y=accuracies, measure='acc')
