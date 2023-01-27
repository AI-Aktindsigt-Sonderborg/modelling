from distutils.util import strtobool

from shared.modelling_utils.input_args import ModellingArgParser


class NERArgParser(ModellingArgParser):
    """
    Class to handle input args for unsupervised Masked Language Modelling
    Methods
    -------
    add_data_params()
        add data parameters
    add_model_params()
        Add model parameters
    add_training_params()
        Add parameters relevant for training - including learning rate scheduling
    add_eval_params()
        Add parameters relevant for evaluation
    add_dp_params()
        Add parameters relevant for differential privacy
    """

    def __init__(self):
        super().__init__()

        self.add_ner_params()

    def add_data_params(self):
        """
        Add data parameters
        """
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument("--train_data", type=str, default='dane',
                                 help="training data file name",
                                 metavar='<str>')
        data_params.add_argument("--eval_data", type=str, default='dane',
                                 help="validation data file name",
                                 metavar='<str>')

    def add_ner_params(self):
        """
        Add model parameters
        """
        ner_params = self.parser.add_argument_group('ner')
        ner_params.add_argument(
            "-lap", "--load_alvenir_pretrained",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=True,
            help="Whether to load local alvenir model")
        ner_params.add_argument(
            "-mn", "--model_name",
            type=str,
            default='last_model',
            help="foundation model from huggingface",
            metavar='<str>')
