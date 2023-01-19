import argparse
from distutils.util import strtobool
from shared.modelling_utils.input_args import ModellingArgParser

class MLMArgParser(ModellingArgParser):
    """
    Class to handle input args for unsupervised Masked Language Modelling
    """
    def __init__(self):
        super().__init__()

        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("-p", '--print_only_args', action='store_true',
                                 help="whether to only print args and exit")

        self.add_mlm_params()

    def add_mlm_params(self):
        mlm_params = self.parser.add_argument_group('mlm')
        mlm_params.add_argument(
            "--whole_word_mask",
            type=lambda x: bool(strtobool(x)),
            default=False,
            metavar='<bool>',
            help="If set to true, whole words are masked")
        mlm_params.add_argument(
            "--mlm_prob",
            type=float,
            default=0.15,
            metavar='<float>',
            help="Probability that a word is replaced by a [MASK] token")
        mlm_params.add_argument(
            "-fl", "--freeze_layers",
            type=lambda x: bool(strtobool(x)),
            default=True,
            metavar='<bool>',
            help="whether to freeze all bert layers until "
                 "freeze_layers_n_steps is reached."
                 "True is mandatory for DP at the moment")
        mlm_params.add_argument(
            "-flns", "--freeze_layers_n_steps",
            type=int,
            default=20000,
            help="number of steps to train head only",
            metavar='<int>')
        mlm_params.add_argument(
            "--replace_head",
            type=lambda x: bool(strtobool(x)),
            default=True,
            help="Whether to replace bert head. True is mandatory for MLM with"
                 " DP - Also set freeze_layers to true if replace_head is true",
            metavar='<bool>')

    def add_data_params(self):
        """
        Add data parameters
        """
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument("--train_data",
                                 type=str,
                                 default='train.json',
                                 help="training data file name",
                                 metavar='<str>')
        data_params.add_argument("--eval_data",
                                 type=str,
                                 default='validation.json',
                                 help="validation data file name",
                                 metavar='<str>')



class SequenceModellingArgParser(ModellingArgParser):
    """
    Class inherited from MLMArgParser to handle input args for supervised
    SequenceClassification
    """
    def add_data_params(self):
        """
        Add data parameters
        """
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument(
            "--train_data",
            type=str,
            default='train_classified.json',
            metavar='<str>',
            help="training data file name")
        data_params.add_argument(
            "--eval_data",
            type=str,
            default='eval_classified.json',
            metavar='<str>',
            help="validation data file name")
        data_params.add_argument(
            "--test_data",
            type=str,
            default='test_classified.json',
            metavar='<str>',
            help="test data file name")



