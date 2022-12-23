import os

from matplotlib import pyplot as plt


def plot_running_results(
        output_dir: str,
        epochs: int,
        lrs,
        accs,
        loss,
        f1,
        epsilon: str = "",
        delta: str = ""):
    """
    Plot results from customized model training
    @param output_dir: directory
    @param epochs: N epochs for title
    @param lrs: Learning rates for plotting
    @param accs: Accuracies for plotting
    @param loss: Losses for plotting
    @param f1: F1 for plotting
    @param epsilon: Epsilon for title
    @param delta: Delta for title
    """
    file_path = os.path.join(output_dir, 'results')
    title = os.path.join(f'Epochs: {epochs}, Epsilon: {epsilon},  Delta: {delta}')
    plt.ioff()
    learning_rate_steps = [int(x['step']) for x in lrs]
    learning_rates = [x['lr'] for x in lrs]

    accuracy_steps = [int(x['step']) for x in accs]
    accuracies = [x['score'] for x in accs]

    losses_steps = [int(x['step']) for x in loss]
    losses = [x['score'] for x in loss]

    f1_steps = [int(x['step']) for x in f1]
    f1s = [x['score'] for x in f1]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle(title, fontsize=14)

    ax1.plot(learning_rate_steps, learning_rates)
    ax1.set(ylabel='learning rate')

    ax2.plot(accuracy_steps, accuracies, 'orange')
    ax2.set(ylabel='accuracy')

    ax3.plot(f1_steps, f1s, 'red')
    ax3.set(ylabel='f1', xlabel='step')

    ax4.plot(losses_steps, losses, 'green')
    ax4.set(ylabel='loss', xlabel='step')
    plt.savefig(file_path)
