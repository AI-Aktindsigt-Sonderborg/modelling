from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

# args.max_length = 128
# args.lot_size = 8


# args.epochs = 5
# args.evaluate_during_training = False

# args.save_config = False
# args.layer_warmup_steps = 100
# args.lr_warmup_steps = 200
args.model_name = 'Geotrend/distilbert-base-da-cased'
args.layer_warmup = False
args.lr_start_decay = 5000
args.lr = 0.00005
args.train_data = 'train_200.json'
# args.save_steps = 20
args.evaluate_steps = 40
args.start_lr = 1
args.end_lr = 1
args.train_batch_size = 2
args.eval_batch_size = 2
args.epochs = 2

mlm_modelling = MLMUnsupervisedModelling(args=args)

mlm_modelling.train_model()
