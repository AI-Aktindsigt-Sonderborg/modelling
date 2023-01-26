from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling, NERModellingDP

ner_parser = NERArgParser()
args = ner_parser.parser.parse_args()

args.local_testing = True
if args.local_testing:
    args.load_alvenir_pretrained = False
    args.replace_head = False
    args.freeze_layers = False
    args.freeze_embeddings = False
    args.differential_privacy = False

    # args.model_name = 'NbAiLab_nb-bert-base-2022-10-19_12-10-45'
    args.model_name = 'NbAiLab/nb-bert-base'
    args.train_batch_size = 2
    args.lot_size = 8
    args.evaluate_batch_size = 2

    args.epochs = 1
    args.evaluate_steps = 50
    args.save_steps = 10000


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
