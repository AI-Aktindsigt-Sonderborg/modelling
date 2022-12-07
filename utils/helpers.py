import json
import os
import time
from typing import Optional, List

import numpy as np
from datasets import load_metric
from opacus.validators import ModuleValidator


def blocks(files, size=65536):
    """
    Read and yield block of lines in file
    :param file: input file(s)
    :param size: max number of lines to read at a time
    :return: yield number of lines
    """
    while True:
        block = files.read(size)
        if not block:
            break
        yield block


def count_num_lines(file_path):
    """
    Count number of lines in file using blocks
    :param file_path: input file path
    :return: number of lines
    """
    with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
        return sum(bl.count("\n") for bl in blocks(file))


def fix_and_validate(model):
    model = ModuleValidator.fix_and_validate(model)
    return model


class TimeCode:
    """
    Class to compute runtime
    """

    def __init__(self):
        self.start_time = time.time()

    def how_long_since_start(self, prefix: Optional[str] = None):
        """
        Compute running time of code since class instantiation
        :param prefix: Optional prefix to print_string
        :return: string with running time in minutes and seconds
        """
        time_end = time.time()
        final_time_seconds = round(time_end - self.start_time, 2)
        final_time_minutes = round(final_time_seconds / 60, 2)
        print_string = f"Time it took: {final_time_seconds} seconds, {final_time_minutes} minutes"
        if prefix:
            print_string = prefix + print_string
        print(print_string)


def append_json(output_dir: str, data: dict, filename: str):
    """
    append json line to jsonlines file
    :param output_dir: directory to save to
    :param data: List[dict]
    :param filename: output file name
    """
    with open(os.path.join(output_dir, filename + '.json'), 'a', encoding='utf-8') as outfile:
        json.dump(data, outfile)
        outfile.write('\n')


def accuracy(preds: np.array, labels: np.array):
    """
    Compute accuracy on predictions and labels
    :param preds: Predicted token
    :param labels: Input labelled token
    :return: Mean accuracy
    """
    return (preds == labels).mean()


def save_json(output_dir: str, data: List[dict], filename: str):
    """
    Save list of dicts to json dump
    :param output_dir:
    :param data: List[dict]
    :param filename: output file name
    """
    with open(os.path.join(output_dir, filename + '.json'), 'w',
              encoding='utf-8') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')

def read_jsonlines(input_dir: str, filename: str):
    """
    read json file to list of dicts
    :param input_dir: directory to read file from
    :param filename: input file name
    """
    data = []
    with open(os.path.join(input_dir, filename + '.json'), 'r',
              encoding='utf-8') as file:
        for line in file:
            data_dict = json.loads(line)
            data.append(data_dict)
    return data

def compute_metrics(eval_pred):
    """
    Computes accuracy on a batch of predictions
    :param eval_pred:
    :return:
    """
    metric = load_metric("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
