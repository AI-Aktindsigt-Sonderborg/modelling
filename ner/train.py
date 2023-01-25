from modelling_utils.ner_modelling import NERModelling, NERModellingDP
from modelling_utils.input_args import NERArgParser

ner_parser = NERArgParser()
args = ner_parser.parser.parse_args()

# args.differential_privacy = False
# args.load_alvenir_pretrained = False
# args.model_name = 'NbAiLab_nb-bert-base-2022-08-11_14-28-23'
if args.differential_privacy:
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        print(ner_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        ner_parser.parser.exit()

    ner_modelling_dp = NERModellingDP(args=args)
    ner_modelling_dp.train_model()
else:
    ner_modelling = NERModelling(args=args)
    ner_modelling.train_model()
