import argparse
from distutils.util import strtobool


class DataPrepArgParser:
    """
    Class to handle input arguments for data preprocessing of scraped data --
    see :class:`.RawScrapePreprocessing`.

    :param int --min_len: Keep sentences with at least n words (default: 8)
    :param float --split: training set size between 0 and 1 (default: 0.98)
    :param bool --add_ppl: Whether to add ppl_score to unique sentences (default: False)
    :param int --ppl_threshold: Perplexity threshold for approving sentences (default: 10000)
    :param str --train_outfile: Name of final training data file fx 'train' (default: train)
    :param str --val_outfile: Name of final validation data file fx 'validation' (default: validation)
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument(
            '--create_bilou',
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to create bilou or use existing")
        self.parser.add_argument(
            '--create_bio_file',
            type=lambda x: bool(strtobool(x)),
            metavar='<bool>',
            default=False,
            help="Whether to create bio file when filtering entities")
        self.parser.add_argument(
            '-oif', '--origin_input_file',
            type=str,
            metavar='<str>',
            default='origin',
            help="original 'raw file' input name")
        self.parser.add_argument(
            '-bif', '--bilou_input_file',
            type=str,
            metavar='<str>',
            default='bilou',
            help="bilou input file name")
        self.parser.add_argument(
            '--split',
            type=float,
            default=0.95, metavar='<float>',
            help='training set size between 0 and 1')
        self.parser.add_argument(
            '--ppl_threshold',
            type=int,
            default=10000,
            metavar='<int>',
            help='ppl_threshold for approving sentences')
        self.parser.add_argument(
            '--train_outfile',
            type=str,
            metavar='<str>',
            default='bilou_train',
            help="Name of final training data file fx 'train'")
        self.parser.add_argument(
            '--val_outfile',
            type=str,
            metavar='<str>',
            default='bilou_val',
            help="Name of final validation data file fx 'validation'")
        self.parser.add_argument(
            '--test_outfile',
            type=str,
            metavar='<str>',
            default='bilou_test',
            help="Name of final test data file fx 'test'")
        self.parser.add_argument(
            "--entities",
            type=str,
            nargs='*',
            default=["PERSON", "LOKATION", "ADRESSE", "HELBRED", "ORGANISATION",
                     "KOMMUNE", "TELEFONNUMMER"],
            metavar='<str>',
            help="define eval metrics to evaluate best model")
        self.parser.add_argument(
            '--print_entity',
            type=str,
            metavar='<str>',
            default=None,
            help="Print specific entity for data inspection.")
        self.parser.add_argument(
            '--test_size',
            type=int,
            metavar='<int>',
            default=25,
            help="Number of sentences containing each class (entity).")
        self.parser.add_argument(
            '--add_dane',
            type=lambda x: bool(strtobool(x)),
            default=True,
            help='whether or not to convert bilou format to bio format used in dane',
            metavar='<bool>')
        self.parser.add_argument(
            '--custom_data_dir', '-cdd',
            type=str,
            metavar='<str>',
            default=None,
            help="Whether to load data from custom directory")
