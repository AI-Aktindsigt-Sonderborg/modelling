from distutils.util import strtobool

from shared.modelling_utils.input_args import ModellingArgParser


class NERArgParser(ModellingArgParser):
    """
    Class inherited from :class:`.ModellingArgParser` to handle input args
    for unsupervised Masked Language Modelling. Below arguments are grouped
    for a better overview.

    *Data*

    :param str --train_data: Training data file name (default: bilou_train.jsonl)
    :param str --eval_data: Validation data file name (default: bilou_val.jsonl)
    :param str --test_data: Validation data file name (default: bilou_test.jsonl)

    *NER Model*

    :param bool --load_alvenir_pretrained: Whether to load local alvenir
        model (default: True)
    :param str --model_name: Foundation model from huggingface or alvenir
        pretrained model (default: sas)
    :param str --data_format: whether to use BILOU or BIO format - input must be either 'bilou' or 'bio' (default: bio)
    :param List[str] --entities: Entities to train on (default: ['PERSON', 'LOKATION', 'ADRESSE', 'HELBRED', 'ORGANISATION', 'KOMMUNE', 'TELEFONNUMMER'])

    """

    def __init__(self):
        super().__init__()

        self.add_ner_params()

    def add_data_params(self):
        """
        Add data parameters
        """
        data_params = self.parser.add_argument_group('data')
        data_params.add_argument("--train_data", type=str, default='bilou_train.jsonl',
                                 help="training data file name",
                                 metavar='<str>')
        data_params.add_argument("--eval_data", type=str, default='bilou_val.jsonl',
                                 help="validation data file name",
                                 metavar='<str>')
        data_params.add_argument("--test_data", type=str, default='bilou_test.jsonl',
                                 help="validation data file name",
                                 metavar='<str>')
        data_params.add_argument("--data_subset", type=int, default=None,
                                 help="Whether to subset data. Must be int "
                                      "between 1 and 100=None. Only relevant "
                                      "while we are training and testing on "
                                      "dane.",
                                 metavar='<int>')

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
            default='sas',
            help="foundation model from huggingface or local if -lap is true",
            metavar='<str>')
        ner_params.add_argument(
            "--entities",
            type=str,
            nargs='*',
            default=["PERSON", "LOKATION", "ADRESSE", "HELBRED", "ORGANISATION", "KOMMUNE", "TELEFONNUMMER"],
            metavar='<str>',
            help="Entities to train on")
        ner_params.add_argument(
            "--concat_bilou",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to compute one single f1_score per entity")
        ner_params.add_argument(
            "-nc", "--normalize_conf",
            type=str,
            default=None,
            help="whether to normalize conf_plot",
            metavar='<str>')
        ner_params.add_argument(
            "-df", "--data_format",
            type=str,
            default="bio",
            help="whether to use BILOU or BIO format - input must be either 'bilou' or 'bio'",
            metavar='<str>')
        ner_params.add_argument(
            "--eval_single",
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to only evaluate one entity per sentence")
