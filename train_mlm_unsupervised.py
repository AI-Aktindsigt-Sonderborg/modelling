from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

# args.local_testing = True

if args.local_testing:
    args.model_name = 'Geotrend/distilbert-base-da-cased'
    args.train_data = 'train_200.json'
    args.evaluate_steps = 100

    args.freeze_layers_n_steps = 20
    args.lr_freezed_warmup_steps = 10
    args.lr_freezed = 0.1

    args.lr_warmup_steps = 10
    args.lr_start_decay = 100
    args.lr = 0.01

    args.epochs = 5
    args.train_batch_size = 2
    args.max_length = 8
    args.save_model_at_end = False
    args.replace_head = True
    args.freeze_layers = True
    # args.evaluate_during_training = False

mlm_modelling = MLMUnsupervisedModelling(args=args)

mlm_modelling.train_model()
