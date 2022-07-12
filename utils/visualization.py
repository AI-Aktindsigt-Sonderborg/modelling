import json
import os
from typing import List

from matplotlib import pyplot as plt

from local_constants import MODEL_DIR


def plot_running_results(output_dir: str, epochs: int, lrs, accs, loss, epsilon: str = "", delta: str = ""):
    file_path = os.path.join(output_dir, 'results')
    title = os.path.join(f'Epochs: {epochs}, Epsilon: {epsilon},  Delta: {delta}')
    plt.ioff()
    learning_rates_steps = [int(x['step']) for x in lrs]
    learning_rates = [x['lr'] for x in lrs]

    accuracies_steps = [int(x['step']) for x in accs]
    accuracies = [x['acc'] for x in accs]

    losses_steps = [int(x['step']) for x in loss]
    losses = [x['loss'] for x in loss]

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(title, fontsize=14)

    ax1.plot(learning_rates_steps, learning_rates)
    ax1.set(ylabel='learning rate')

    ax2.plot(accuracies_steps, accuracies, 'orange')
    ax2.set(ylabel='accuracy')

    ax3.plot(losses_steps, losses, 'green')
    ax3.set(ylabel='loss', xlabel='step')
    plt.savefig(file_path)
    # plt.close(fig=fig)

    # plt.show()


def plot_performance(x: List[int], y: List[float], measure: str):
    plt.title('loss')
    plt.xlabel('step')
    plt.legend(measure)
    plt.plot(x, y)

    plt.show()


if __name__ == '__main__':

    model_name = 'DP-Geotrend_distilbert-base-da-cased-2022-06-28_14-27-19'

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

    plot_running_results(output_path='../plots/test2.png', lrs=lrs, accs=accs, loss=loss_init)

    # plot_performance(x=losses_steps, y=losses, measure='loss')
    # plot_performance(x=accuracies_steps, y=accuracies, measure='acc')
