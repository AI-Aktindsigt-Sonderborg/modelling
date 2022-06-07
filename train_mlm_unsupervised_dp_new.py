from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

args.model_name = 'Geotrend/distilbert-base-da-cased'
# args.evaluate_during_training = False
# args.save_config = False

mlm_modelling = MLMUnsupervisedModelling(args=args)


mlm_modelling.preprocess_and_train_model()
