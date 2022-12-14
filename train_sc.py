"""Script to train a sequence-classification model with custom train loop"""

from modelling_utils.input_args import SequenceModellingArgParser
from modelling_utils.sequence_classification import SequenceClassificationDP, SequenceClassification

sc_parser = SequenceModellingArgParser()
args = sc_parser.parser.parse_args()

label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhverv og turisme': 2,
              'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)
args.labels = LABELS
# args.load_alvenir_pretrained = False
# args.model_name = 'NbAiLab/nb-bert-base'
args.differential_privacy = False
args.local_testing = False

if args.local_testing:
    args.train_batch_size = 4
    args.eval_batch_size = 4
    args.evaluate_steps = 50

if not ((args.lot_size > args.train_batch_size) and (args.lot_size % args.train_batch_size == 0)):
    print(sc_parser.parser._option_string_actions['--lot_size'].help)
    print('exiting - try again')
    sc_parser.parser.exit()

if args.differential_privacy:
    sc_modelling = SequenceClassificationDP(args=args)
else:
    sc_modelling = SequenceClassification(args=args)

sc_modelling.train_model()
