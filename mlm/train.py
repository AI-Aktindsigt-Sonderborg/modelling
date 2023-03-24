# pylint: disable=protected-access, broad-except
"""Main script to train a Masked Language Model with custom train
loop"""
import sys
import traceback

from mlm.modelling_utils.input_args import MLMArgParser
from mlm.modelling_utils.mlm_modelling import MLMModelling, MLMModellingDP
from shared.utils.helpers import init_logging

logger = init_logging(model_type='MLM', log_path='logs/model_log.log')

# argument parser for MLM model
mlm_parser = MLMArgParser()

args, leftovers = mlm_parser.parser.parse_known_args()
if leftovers:
    model_name_to_print = args.custom_model_name if \
        args.custom_model_name else args.model_name
    logger.warning(f'The following args is not relevant for model '
                   f'{model_name_to_print}: '
                   f'{leftovers}. Ignoring...')

args.cmd_line_args = sys.argv

if args.replace_head is True and args.freeze_layers is False:
    print(mlm_parser.parser._option_string_actions['--replace_head'].help)
    print('exiting - try again')
    mlm_parser.parser.exit()

if args.freeze_layers and args.freeze_layers_n_steps == 0:
    print(mlm_parser.parser._option_string_actions['--freeze_layers_n_steps'])
    print('exiting - try again')
    mlm_parser.parser.exit()

if args.differential_privacy:
    mlm_modelling = MLMModellingDP(args=args)
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        logger.warning(
            f'Model: {mlm_modelling.args.output_name} - '
            f'{mlm_parser.parser._option_string_actions["--lot_size"].help}')
        print(mlm_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()
    elif not (args.replace_head is True and args.freeze_layers is True):
        logger.warning(
            f'DP model: {mlm_modelling.args.output_name} is training without'
            f' replacing head and freezing layers.')
    elif not args.freeze_embeddings:
        logger.error(
            f'Model: {mlm_modelling.args.output_name} - '
            f'{mlm_parser.parser._option_string_actions["--freeze_embeddings"].help}')
        print(mlm_parser.parser._option_string_actions[
                  '--freeze_embeddings'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()

else:
    args.epsilon = None
    args.delta = None
    args.compute_delta = None
    args.max_grad_norm = None
    mlm_modelling = MLMModelling(args=args)

try:
    logger.info(f"Training model {mlm_modelling.args.output_name}")
    mlm_modelling.train_model()
    logger.info(
        f'Model {mlm_modelling.args.output_name} trained succesfully')
except Exception as ex:
    logger.error(
        f'Model {mlm_modelling.args.output_name} failed:\n{traceback.format_exc()}')
