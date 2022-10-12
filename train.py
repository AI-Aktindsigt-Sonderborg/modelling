from modelling_utils.mlm_modelling import MLMUnsupervisedModelling, MLMUnsupervisedModellingDP
from utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

if args.differential_privacy:
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        print(mlm_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()

    mlm_modelling_dp = MLMUnsupervisedModellingDP(args=args)
    mlm_modelling_dp.train_model()
else:
    mlm_modelling = MLMUnsupervisedModelling(args=args)
    mlm_modelling.train_model()
