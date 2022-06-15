from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()
# args.model_name = 'vesteinn/ScandiBERT'
# args.epochs = 5
# args.max_length = 128
# args.lot_size = 8
# args.save_steps = 20
# args.model_name = 'Geotrend/distilbert-base-da-cased'
# args.train_data = 'train_200.json'
# args.save_config = False
# args.warmup = True
# args.warmup_steps = 100

mlm_modelling = MLMUnsupervisedModelling(args=args)


mlm_modelling.train_model()
