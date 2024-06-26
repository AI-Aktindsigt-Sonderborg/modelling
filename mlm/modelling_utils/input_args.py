from distutils.util import strtobool

from shared.modelling_utils.input_args import ModellingArgParser


class MLMArgParser(ModellingArgParser):
    """
    Class inherited from :class:`.ModellingArgParser` to handle input args for
    unsupervised Masked Language Modelling. Below arguments are grouped for a
    better overview.

    *Data*

    :param str --train_data: Training data file name (default: train.jsonl)
    :param str --eval_data: Validation data file name (default: validation.jsonl)
    :param bool --load_alvenir_pretrained: Whether to load local alvenir
        model (default: False)

    *MLM Model*

    :param --model_name: Foundation model from huggingface (default:
        NbAiLab/nb-bert-base)
    :param bool --whole_word_mask: If set to true, whole words are masked (
        default: False)
    :param float --mlm_prob: Probability that a word is replaced by a [MASK]
        token (default:0.15)
    :param bool --replace_head: Whether to replace bert head. True is
        mandatory for MLM with DP - Also set freeze_layers to true if
        replace_head is true (default: True)

    """

    def __init__(self):
        super().__init__()

        self.add_mlm_params()

    def add_mlm_params(self):
        mlm_params = self.parser.add_argument_group('mlm')
        mlm_params.add_argument(
            "-lap", "--load_alvenir_pretrained",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to load local alvenir model")
        mlm_params.add_argument(
            "-mn", "--model_name",
            type=str,
            default='NbAiLab/nb-bert-base',
            help="foundation model from huggingface",
            metavar='<str>')
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
            "-rh","--replace_head",
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
                                 default='train.jsonl',
                                 help="training data file name",
                                 metavar='<str>')
        data_params.add_argument("--eval_data",
                                 type=str,
                                 default='validation.jsonl',
                                 help="validation data file name",
                                 metavar='<str>')
