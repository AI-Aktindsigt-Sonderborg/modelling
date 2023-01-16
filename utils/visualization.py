import os
from typing import List

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from data_utils.custom_dataclasses import EvalScore


def plot_running_results(
        output_dir: str,
        epochs: int,
        metrics: List[EvalScore],
        lrs,
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

    metric_steps = [x.step for x in metrics]
    accuracies = [x.accuracy for x in metrics]
    losses = [x.loss for x in metrics]
    f_1s = [x.f_1 for x in metrics]

    learning_rate_steps = [int(x['step']) for x in lrs]
    learning_rates = [x['lr'] for x in lrs]

    # accuracy_steps = [int(x['step']) for x in accs]
    # accuracies = [x['score'] for x in accs]

    # losses_steps = [int(x['step']) for x in loss]
    # losses = [x['score'] for x in loss]

    # f1_steps = [int(x['step']) for x in f1]
    # f1s = [x['score'] for x in f1]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle(title, fontsize=14)

    ax1.plot(learning_rate_steps, learning_rates)
    ax1.set(ylabel='learning rate')

    ax2.plot(metric_steps, accuracies, 'orange')
    ax2.set(ylabel='accuracy')

    ax3.plot(metric_steps, f_1s, 'red')
    ax3.set(ylabel='f1', xlabel='step')

    ax4.plot(metric_steps, losses, 'green')
    ax4.set(ylabel='loss', xlabel='step')
    plt.tight_layout()
    plt.savefig(file_path)


def plot_confusion_matrix(
        y_true,
        y_pred,
        labels,
        model_name: str,
        normalize: str = 'true',
        save_fig: bool = True):
    """Function is self-explanatory"""

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels,
                                   normalize=normalize)
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g', xticklabels=labels,
               yticklabels=labels)

    plt.tight_layout()
    if save_fig:
        plt.savefig(f'plots/conf_plot_{model_name.replace("/", "_")}')
    else:
        plt.show()


def plot_pca_for_demo(embedding_outputs, modelling):
    X = np.array([x.embedding for x in embedding_outputs])
    y_true = [int(modelling.label2id[x.label]) for x in embedding_outputs]
    y_pred = [int(modelling.label2id[x.prediction]) for x in embedding_outputs]

    sc = StandardScaler(with_mean=False)
    sc.fit(X)
    X_std = sc.transform(X)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_std)

    exp_var_pca = pca.explained_variance_ratio_

    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues,
             where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_true, cmap='rainbow')
    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')
    # ax.set_zlabel('PCA 3')
    # plt.title('True labels')
    # plt.show()
    #
    # # Create a 3D scatter plot of the PCA transformed data
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_pred, cmap='rainbow')
    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')
    # ax.set_zlabel('PCA 3')
    # plt.title('Predicted labels')
    # plt.show()

    # Create a scatter plot of the PCA transformed data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='rainbow')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('True labels')
    plt.show()

    # Create a scatter plot of the PCA transformed data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='rainbow')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Predicted labels')
    plt.show()
