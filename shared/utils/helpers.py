import json
import logging
import os
import time
from typing import Optional, List

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


def init_logging(model_type: str, log_path: str):
    """
    Method to set up log file to write logs
    :param model_type: model of type either NER, SC or MLM
    :param log_path: path to log file
    :return: logger
    """
    logger = logging.getLogger(f'{model_type} model log')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


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
        print_string = f"Time it took: {final_time_seconds} seconds," \
                       f" {final_time_minutes} minutes"
        if prefix:
            print_string = prefix + print_string
        print(print_string)


def append_json_lines(output_dir: str, data: dict, filename: str,
                extension: str = '.jsonl'):
    """
    append json line to jsonlines file
    :param output_dir: directory to save to
    :param data: List[dict]
    :param filename: output file name
    """
    with open(os.path.join(output_dir, filename + extension), 'a',
              encoding='utf-8') as outfile:
        json.dump(data, outfile)
        outfile.write('\n')

def write_text_lines(out_dir: str, filename: str, data: List[str]):
    """
    Write text file based on list of strings
    :param out_dir: directory
    :param filename: filename to write
    :param data: input data
    """
    with open(os.path.join(out_dir, f'{filename}.txt'), 'w',
              encoding='utf-8') as outfile:
        for entry in data:
            outfile.write(f"{entry}\n")

def read_json_lines(input_dir: str, filename: str, extension: str = '.jsonl'):
    """
    read json file to list of dicts
    :param input_dir: directory to read file from
    :param filename: input file name
    """
    data = []
    with open(os.path.join(input_dir, filename + extension), 'r',
              encoding='utf-8') as file:
        for line in file:
            data_dict = json.loads(line)
            data.append(data_dict)
    return data

def write_json_lines(out_dir: str, filename: str, data: List[dict],
                     extension: str = '.jsonl'):
    """
    Write json_lines_file based on list of dictionaries
    :param out_dir: directory
    :param filename: filename to write
    :param data: input data
    :param extension: file extension
    """
    with open(os.path.join(out_dir, filename + extension), 'w',
              encoding='utf-8') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')
    print(f'Created file {os.path.join(out_dir, filename + extension)}')

def read_json(filepath: str):
    """
    read json file to list of dicts
    :param input_dir: directory to read file from
    :param filename: input file name
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
