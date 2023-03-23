# pylint: disable=protected-access
"""Main script to train a ner model with custom train loop"""
import logging
import sys

from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling, NERModellingDP

logger = logging.getLogger('MLM model log')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('model_log.log')
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

# argument parser for NER model
ner_parser = NERArgParser()

args, leftovers = ner_parser.parser.parse_known_args()
if leftovers:
    logger.warning(f'The following args is not relevant for this model: '
                   f'{leftovers}')

args.cmd_line_args = sys.argv

if args.differential_privacy:
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        print(ner_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        ner_parser.parser.exit()
    elif not args.freeze_embeddings:
        print(ner_parser.parser._option_string_actions[
                  '--freeze_embeddings'].help)
        print('exiting - try again')
        ner_parser.parser.exit()

    ner_modelling_dp = NERModellingDP(args=args)
    ner_modelling_dp.train_model()
else:
    ner_modelling = NERModelling(args=args)
    ner_modelling.train_model()
