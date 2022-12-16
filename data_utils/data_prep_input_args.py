import argparse
from distutils.util import strtobool

class DataPrepArgParser:
    """
    Class to handle arguments of data prep for scraped data
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--save_data', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether or not to add ppl_score to unique sentences',
                                 metavar='<bool>')
        self.parser.add_argument('--danish_threshold', type=float, default=0.6, metavar='<float>',
                        help='threshold to filter danish sentence predictions from raw scrape')
        self.parser.add_argument('--min_len', type=int, default=8, metavar='<int>',
                        help='Keep sentences with at least n words')
        self.parser.add_argument('--split', type=float, default=0.98, metavar='<float>',
                        help='training set size between 0 and 1')
        self.parser.add_argument('--add_ppl', type=lambda x: bool(strtobool(x)), default=True,
                        help='whether or not to add ppl_score to unique sentences',
                                 metavar='<bool>')
        self.parser.add_argument('--ppl_threshold', type=int, default=10000, metavar='<int>',
                        help='ppl_threshold for approving sentences')
        self.parser.add_argument('--train_outfile', type=str, metavar='<str>',
                        default='train', help="Name of final training data file")
        self.parser.add_argument('-stnt', '--split_train_n_times', type=int, metavar='<int>',
                                 default=0, help="Split train set n times into n+1 training sets")
        self.parser.add_argument('--val_outfile', type=str, metavar='<str>',
                        default='validation', help="Name of final validation data file")
        self.parser.add_argument('--excel_classification_file', type=str, metavar='<str>',
                                 default='skrab_01.xlsx', help="name of annotated excel file")
        self.parser.add_argument('--classified_scrape_file', type=str, metavar='<str>',
                                 default='classified_scrape1', help="name of classified scrape file")
