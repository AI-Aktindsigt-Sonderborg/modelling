from distutils.util import strtobool

from shared.modelling_utils.input_args import ModellingArgParser

class SequenceModellingArgParser(ModellingArgParser):
    """
    Class inherited from MLMArgParser to handle input args for supervised
    SequenceClassification
data:
    :param str --train_data: Training data file name (default:
    train_classified.jsonl)
    :param str --eval_data: Validation data file name (default:
    eval_classified.jsonl)
    :param str --test_data: Validation data file name (default:
    test_classified.jsonl)

seq_class:
    :param bool --load_alvenir_pretrained: Whether to load local alvenir
    model (default: True)
    :param --model_name: Foundation model from huggingface (default: last_model)
    """

    def __init__(self):
        super().__init__()
        self.add_sc_params()

    def add_data_params(self):
        """
        Add data parameters
        """
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument(
            "--train_data",
            type=str,
            default='train_classified.jsonl',
            metavar='<str>',
            help="training data file name")
        data_params.add_argument(
            "--eval_data",
            type=str,
            default='eval_classified.jsonl',
            metavar='<str>',
            help="validation data file name")
        data_params.add_argument(
            "--test_data",
            type=str,
            default='test_classified.jsonl',
            metavar='<str>',
            help="test data file name")

    def add_sc_params(self):
        sc_params = self.parser.add_argument_group('seq_class')
        sc_params.add_argument(
            "-lap", "--load_alvenir_pretrained",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=True,
            help="Whether to load local alvenir model")
        sc_params.add_argument(
            "-mn", "--model_name",
            type=str,
            default='last_model',
            help="foundation model from huggingface",
            metavar='<str>')
