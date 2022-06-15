from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

# args.model_name = 'Geotrend/distilbert-base-da-cased'
# args.model_name = 'vesteinn/ScandiBERT'
# args.evaluate_during_training = True
# args.train_data = 'train_200.json'
# args.epochs = 5
# args.save_config = False
# args.max_length = 128
# args.lot_size = 8
mlm_modelling = MLMUnsupervisedModelling(args=args)


mlm_modelling.train_model()


print()