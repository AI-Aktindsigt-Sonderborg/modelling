import sys

from mlm.modelling_utils.mlm_modelling import MLMModelling, MLMModellingDP
from mlm.modelling_utils.input_args import MLMArgParser

mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()
args.cmd_line_args = sys.argv

if args.replace_head is True and args.freeze_layers is False:
    print(mlm_parser.parser._option_string_actions['--replace_head'].help)
    print('exiting - try again')
    mlm_parser.parser.exit()

if args.differential_privacy:
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        print(mlm_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()
    elif not (args.replace_head is True and args.freeze_layers is True
              and args.freeze_embeddings is True):
        print(mlm_parser.parser._option_string_actions[
                  '--freeze_embeddings'].help)
        print('exiting - try again')
        mlm_parser.parser.exit()

    mlm_modelling_dp = MLMModellingDP(args=args)
    mlm_modelling_dp.train_model()
else:
    args.epsilon = None
    args.delta = None
    args.compute_delta = None
    args.max_grad_norm = None
    mlm_modelling = MLMModelling(args=args)
    mlm_modelling.train_model()
