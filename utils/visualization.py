import os

from matplotlib import pyplot as plt


def plot_running_results(output_dir: str, epochs: int, lrs, accs, loss, epsilon: str = "",
                         delta: str = ""):
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
