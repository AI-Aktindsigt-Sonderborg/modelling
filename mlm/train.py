# pylint: disable=protected-access
"""Main script to train a Masked Language Model with custom train
loop"""
import logging
import sys

from mlm.modelling_utils.mlm_modelling import MLMModelling, MLMModellingDP
from mlm.modelling_utils.input_args import MLMArgParser

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

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

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
    # try:
    #     logger.info(f"Training model {mlm_modelling.args.output_name}")
    #     mlm_modelling.train_model()
    #     logger.info(
    #         f'Model {mlm_modelling.args.output_name} trained succesfully')
    #     # print(f'\033[94m\nModel {mlm_modelling.args.output_name}'
    #     #       f'was trained with great success\033[0m')
    # except Exception as ex:
    #     logger.error(f'Model {mlm_modelling.args.output_name} failed:\n{ex}')

try:
    logger.info(f"Training model {mlm_modelling.args.output_name}")
    mlm_modelling.train_model()
    logger.info(
        f'Model {mlm_modelling.args.output_name} trained succesfully')
except Exception as ex:
    logger.error(f'Model {mlm_modelling.args.output_name} failed:\n{ex}')
