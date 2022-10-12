# pylint: disable=protected-access
import torch

from modelling_utils.mlm_modelling import MLMUnsupervisedModellingDP, MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

# to check if cuda available - if not update pycharm fx
# import torch
# a = torch.cuda.is_available()

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

args.local_testing = True

# hardcode these two as they are essential for DP training atm
args.freeze_layers = True
args.replace_head = True

if args.local_testing:
    args.model_name = 'Geotrend/distilbert-base-da-cased'
    # args.model_name = 'NbAiLab_nb-bert-base-2022-08-11_14-28-23'
    # args.model_name = 'NbAiLab/nb-bert-base'
    args.train_data = 'train_10.json'
    args.eval_data = 'val_10.json'
    args.evaluate_steps = 2
    args.logging_steps = 2
    args.save_steps = 2
    args.freeze_layers_n_steps = 20
    args.lr_freezed_warmup_steps = 10
    args.lr_freezed = 0.001
    args.learning_rate = 0.001
    args.lr_warmup_steps = 10
    args.lr_start_decay = 20
    # args.lr = 0.01

    args.epochs = 5
    args.train_batch_size = 2
    args.max_length = 8
    # args.save_model_at_end = False
    args.make_plots = True
    args.dp = False
    args.freeze_layers = True
    # args.simulate_batches = True
    # args.batch_multiplier = 2
    args.load_alvenir_pretrained = False
    # args.freeze_embeddings = False
    args.freeze_layers_n_steps = 2
    # args.p = True


if not ((args.lot_size > args.train_batch_size) and (args.lot_size % args.train_batch_size == 0)):
    print(mlm_parser.parser._option_string_actions['--lot_size'].help)
    print('exiting - try again')
    mlm_parser.parser.exit()

if args.differential_privacy:
    mlm_modelling = MLMUnsupervisedModellingDP(args=args)
else:
    mlm_modelling = MLMUnsupervisedModelling(args=args)

mlm_modelling.train_model()


# for name, param in model.named_parameters():
#     if 'embed' in name:
#         print(f'name: {name}, param: {param.requires_grad}')