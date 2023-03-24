# pylint: disable=protected-access, broad-except
"""Main script to train a ner model with custom train loop"""
import sys
import traceback

from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling, NERModellingDP
from shared.utils.helpers import init_logging

logger = init_logging(model_type='NER', log_path='logs/model_log.log')

# argument parser for NER model
ner_parser = NERArgParser()

args, leftovers = ner_parser.parser.parse_known_args()

model_name_to_print = args.custom_model_name if \
    args.custom_model_name else args.model_name
if leftovers:
    logger.warning(f'The following args is not relevant for model '
                   f'{model_name_to_print}: '
                   f'{leftovers}. Ignoring...')

if args.freeze_layers:
    logger.warning(
        f'Freezing layers for model {model_name_to_print} has not been '
        f'implemented')
    args.freeze_layers = False

args.cmd_line_args = sys.argv

if args.differential_privacy:
    ner_modelling = NERModellingDP(args=args)
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        logger.warning(
            f'Model: {ner_modelling.args.output_name} - '
            f'{ner_parser.parser._option_string_actions["--lot_size"].help}')
        print('exiting - try again')
        ner_parser.parser.exit()
    elif not args.freeze_embeddings:
        logger.error(
            f'Model: {ner_modelling.args.output_name} - '
            f'{ner_parser.parser._option_string_actions["--freeze_embeddings"].help}')
        print(ner_parser.parser._option_string_actions[
                  '--freeze_embeddings'].help)
        print('exiting - try again')
        ner_parser.parser.exit()

else:
    ner_modelling = NERModelling(args=args)

try:
    logger.info(f"Training model {ner_modelling.args.output_name}")
    ner_modelling.train_model()
    logger.info(
        f'Model {ner_modelling.args.output_name} trained succesfully')
except Exception as ex:
    logger.error(
        f'Model {ner_modelling.args.output_name} '
        f'failed:\n{traceback.format_exc()}')
