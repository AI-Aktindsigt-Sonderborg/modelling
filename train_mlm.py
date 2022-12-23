
from modelling_utils.mlm_modelling import MLMModelling, MLMModellingDP
from modelling_utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

if not (args.replace_head == True and args.freeze_layers == True):
    print(mlm_parser.parser._option_string_actions['--replace_head'].help)
    print('exiting - try again')
    mlm_parser.parser.exit()

if args.differential_privacy:
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        print(mlm_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()
    elif not (args.replace_head == True and args.freeze_layers == True
              and args.freeze_embeddings):
        print(mlm_parser.parser._option_string_actions['--freeze_embeddings'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()

    mlm_modelling_dp = MLMModellingDP(args=args)
    mlm_modelling_dp.train_model()
else:
    mlm_modelling = MLMModelling(args=args)
    mlm_modelling.train_model()
