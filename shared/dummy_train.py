# pylint: disable=protected-access, broad-except
"""Main script to train a Masked Language Model with custom train
loop"""
import sys
import traceback

from mlm.modelling_utils.input_args import ModellingArgParser
from shared.modelling_utils.modelling import Modelling
from shared.utils.helpers import init_logging

logger = init_logging(model_type='default', log_path='logs/model_log.log')

# argument parser for MLM model
arg_parser = ModellingArgParser()

args, leftovers = arg_parser.parser.parse_known_args()
if leftovers:
    model_name_to_print = args.custom_model_name if \
        args.custom_model_name else args.model_name
    logger.warning(f'The following args is not relevant for model '
                   f'{model_name_to_print}: '
                   f'{leftovers}. Ignoring...')

args.cmd_line_args = sys.argv



modelling = Modelling(args=args)

try:
    logger.info(f"Training model {modelling.args.output_name}")
    modelling.train_model()
    logger.info(
        f'Model {modelling.args.output_name} trained succesfully')
except Exception as ex:
    logger.error(
        f'Model {modelling.args.output_name} '
        f'failed:\n{traceback.format_exc()}')
