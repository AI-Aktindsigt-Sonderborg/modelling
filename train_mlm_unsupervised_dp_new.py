from modelling_utils.mlm_modelling import MLMUnsupervisedModelling, MLMUnsupervisedModellingDP
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()



# args.local_testing = True

if args.local_testing:
    args.model_name = 'Geotrend/distilbert-base-da-cased'
    args.train_data = 'train_200.json'
    args.evaluate_steps = 20
    args.save_config = False
    args.layer_warmup_steps = 100
    args.lr_warmup_steps = 200
    args.lr_start_decay = 300
    args.epochs = 5
    args.train_batch_size = 16
    args.max_length = 8


mlm_modelling_dp = MLMUnsupervisedModellingDP(args=args)

mlm_modelling_dp.train_model()
