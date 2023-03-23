# pylint: disable=protected-access
"""Main script to train a sequence-classification model with custom train
loop"""
import logging
import sys

from sc.modelling_utils.input_args import SequenceModellingArgParser
from sc.modelling_utils.sequence_classification import \
    SequenceClassificationDP, SequenceClassification

logger = logging.getLogger('SC model log')
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

# argument parser for SC model
sc_parser = SequenceModellingArgParser()

args, leftovers = sc_parser.parser.parse_known_args()
if leftovers:
    logger.warning(f'The following args is not relevant for this model: '
                   f'{leftovers}.. ignoring')

args.cmd_line_args = sys.argv

label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
              'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)
args.labels = LABELS

if args.differential_privacy:
    sc_modelling = SequenceClassificationDP(args=args)
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        logger.warning(
            f'Model: {sc_modelling.args.output_name} - '
            f'{sc_parser.parser._option_string_actions["--lot_size"].help}')

        print(sc_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        sc_parser.parser.exit()
    elif not args.freeze_embeddings:
        print(sc_parser.parser._option_string_actions[
                  '--freeze_embeddings'].help)
        logger.error(
            f'Model: {sc_modelling.args.output_name} - '
            f'{sc_parser.parser._option_string_actions["--freeze_embeddings"].help}')
        print('exiting - try again')
        sc_parser.parser.exit()

else:
    sc_modelling = SequenceClassification(args=args)

try:
    logger.info(f"Training model {sc_modelling.args.output_name}")
    sc_modelling.train_model()
    logger.info(
        f'Model {sc_modelling.args.output_name} trained succesfully')
except Exception as ex:
    logger.error(f'Model {sc_modelling.args.output_name} failed:\n{ex}')

