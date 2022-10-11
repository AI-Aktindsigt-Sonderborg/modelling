import argparse
from distutils.util import strtobool

class DataPrepArgParser:
    def __init__(self):

        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--save_data', type=lambda x: bool(strtobool(x)), default=False,
                        help='whether or not to add ppl_score to unique sentences')
        self.parser.add_argument('--danish_threshold', type=float, default=0.6,
                        help='threshold to filter danish sentence predictions from raw scrape')
        self.parser.add_argument('--min_len', type=int, default=8,
                        help='Keep sentences with at least n words')
        self.parser.add_argument('--split', type=float, default=0.98,
                        help='training set size between 0 and 1')
        self.parser.add_argument('--add_ppl', type=lambda x: bool(strtobool(x)), default=True,
                        help='whether or not to add ppl_score to unique sentences')
        self.parser.add_argument('--ppl_threshold', type=int, default=10000,
                        help='ppl_threshold for approving sentences')

        self.parser.add_argument('--train_outfile', type=str,
                        default='train', help="Name of final training data file")
        self.parser.add_argument('--val_outfile', type=str,
                        default='validation', help="Name of final validation data file")
