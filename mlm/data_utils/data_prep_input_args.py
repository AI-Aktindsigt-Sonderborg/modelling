import argparse
from distutils.util import strtobool


class DataPrepArgParser:
    """
    Class to handle input arguments for data preprocessing of scraped data --
    see :class:`.RawScrapePreprocessing`.

    :param  float --danish_threshold: Threshold to filter danish sentence predictions from raw scrape (default: 0.6)
    :param int --min_len: Keep sentences with at least n words (default: 8)
    :param float --split: training set size between 0 and 1 (default: 0.98)
    :param bool --add_ppl: Whether to add ppl_score to unique sentences (default: False)
    :param int --ppl_threshold: Perplexity threshold for approving sentences (default: 10000)
    :param str --train_outfile: Name of final training data file fx 'train' (default: train)
    :param str --val_outfile: Name of final validation data file fx 'validation' (default: validation)
    :param int --split_train_n_times: Split train set n times into n+1 training sets (default: 0)
    :param str --excel_classification_file: Name of annotated excel file (default: skrab_01.xlsx)
    :param str --classified_scrape_file: Name of classified scrape file (default: classified_scrape)
    :param bool --lower_case: Whether to lower case all sentences. If doing mlm,
        very important to check whether classified data is lowercased (default: False)
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument(
            '--danish_threshold',
            type=float,
            default=0.6,
            metavar='<float>',
            help='threshold to filter danish sentence predictions from raw '
                 'scrape')
        self.parser.add_argument(
            '--min_len',
            type=int,
            default=8,
            metavar='<int>',
            help='Keep sentences with at least n words')
        self.parser.add_argument(
            '--split',
            type=float,
            default=0.98, metavar='<float>',
            help='training set size between 0 and 1')
        self.parser.add_argument(
            '--add_ppl',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='whether or not to add ppl_score to unique sentences',
            metavar='<bool>')
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
            default='train',
            help="Name of final training data file fx 'train'")
        self.parser.add_argument(
            '--val_outfile',
            type=str,
            metavar='<str>',
            default='validation',
            help="Name of final validation data file fx 'validation'")
        self.parser.add_argument(
            '-stnt', '--split_train_n_times',
            type=int,
            metavar='<int>',
            default=0,
            help="Split train set n times into n+1 training sets")
        self.parser.add_argument(
            '--excel_classification_file',
            type=str,
            metavar='<str>',
            default='skrab_01.xlsx',
            help="name of annotated excel file")
        self.parser.add_argument(
            '--classified_scrape_file',
            type=str,
            metavar='<str>',
            default='classified_scrape',
            help="name of classified scrape file")
        self.parser.add_argument(
            '--lower_case',
            type=lambda x: bool(strtobool(x)),
            default=False,
            help='whether or not to lower case all sentences. '
                 'If doing mlm, very important to check whether classified data '
                 'is lowercased',
            metavar='<bool>')
        self.parser.add_argument(
            '--raw_file',
            type=str,
            metavar='<str>',
            default='raw1',
            help="raw file name")
        self.parser.add_argument(
            '--sc_class_size',
            type=int,
            metavar='<int>',
            default=1400,
            help="Number of sentences within each class used for SequenceClassification.")
