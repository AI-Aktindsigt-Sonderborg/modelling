# pylint: disable=too-many-locals
import os
from typing import List

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score
from shared.data_utils.custom_dataclasses import EvalScore


def plot_running_results(
    output_dir: str,
    epochs: int,
    metrics: List[EvalScore],
    lrs,
    epsilon: str = "",
    delta: str = "",
):
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
    file_path = os.path.join(output_dir, "results")
    title = os.path.join(f"Epochs: {epochs}, Epsilon: {epsilon},  Delta: {delta}")
    plt.ioff()

    metric_steps = [x.step for x in metrics]
    accuracies = [x.accuracy for x in metrics]
    losses = [x.loss for x in metrics]
    f_1s = [x.f_1 for x in metrics]

    learning_rate_steps = [int(x["step"]) for x in lrs]
    learning_rates = [x["lr"] for x in lrs]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle(title, fontsize=14)

    ax1.plot(learning_rate_steps, learning_rates)
    ax1.set(ylabel="learning rate")

    ax2.plot(metric_steps, accuracies, "orange")
    ax2.set(ylabel="accuracy")

    ax3.plot(metric_steps, f_1s, "red")
    ax3.set(ylabel="f1", xlabel="step")

    ax4.plot(metric_steps, losses, "green")
    ax4.set(ylabel="loss", xlabel="step")
    plt.tight_layout()
    plt.savefig(file_path)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    model_name: str,
    plots_dir: str = None,
    normalize: str = "true",
    save_fig: bool = True,
    concat_bilou: bool = False,
):
    """Function is self-explanatory"""

    if concat_bilou and "ner" in plots_dir:
        y_true = [y[2:] if y != "O" else y for y in y_true]
        y_pred = [y[2:] if y != "O" else y for y in y_pred]
        labels = list(
            OrderedDict.fromkeys(
                [label[2:] if label != "O" else label for label in labels]
            )
        )

        print(
            f"eval precision concat: {precision_score(y_true, y_pred, average='macro')}"
        )
        print(f"eval recall concat: {recall_score(y_true, y_pred, average='macro')}")
        print(f"eval f1 concat: {f1_score(y_true, y_pred, average='macro')}")

    # print(labels)
    # print(y_true)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(20, 14))
    sn.heatmap(
        df_cm,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    if save_fig:
        filepath = os.path.join(plots_dir, f'conf_plot_{model_name.replace("/", "_")}')
        if concat_bilou:
            filepath += "-concat_bilou"
        plt.savefig(filepath)
    else:
        plt.show()
