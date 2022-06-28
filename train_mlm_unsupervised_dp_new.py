import sys

from modelling_utils.mlm_modelling import MLMUnsupervisedModelling, MLMUnsupervisedModellingDP
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

# args.local_testing = True

if args.local_testing:
    args.model_name = 'Geotrend/distilbert-base-da-cased'
    args.train_data = 'train_200.json'
    args.evaluate_steps = 200
    args.save_config = False


    args.freeze_layers_n_steps = 20
    args.lr_freezed_warmup_steps = 10
    args.lr_freezed = 0.1

    args.lr_warmup_steps = 10
    args.lr_start_decay = 100
    args.lr = 0.01

    args.epochs = 5
    args.train_batch_size = 2
    args.max_length = 8


if not ((args.lot_size > args.train_batch_size) and (args.lot_size % args.train_batch_size == 0)):
    print(mlm_parser.parser._option_string_actions['--lot_size'].help)
    print('exiting - try again')
    mlm_parser.parser.exit()



mlm_modelling_dp = MLMUnsupervisedModellingDP(args=args)

mlm_modelling_dp.train_model()
