# pylint: disable=protected-access
"""Main script to train a sequence-classification model with custom train
loop"""
import sys

from sc.modelling_utils.input_args import SequenceModellingArgParser
from sc.modelling_utils.sequence_classification import \
    SequenceClassificationDP, SequenceClassification

sc_parser = SequenceModellingArgParser()
args = sc_parser.parser.parse_args()
args.cmd_line_args = sys.argv

label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
              'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)
args.labels = LABELS

if args.differential_privacy:
    if not ((args.lot_size > args.train_batch_size)
            and (args.lot_size % args.train_batch_size == 0)):
        print(sc_parser.parser._option_string_actions['--lot_size'].help)
        print('exiting - try again')
        sc_parser.parser.exit()
    elif not args.freeze_embeddings:
        print(sc_parser.parser._option_string_actions[
                  '--freeze_embeddings'].help)
        print('exiting - try again')
        sc_parser.parser.exit()

    sc_modelling = SequenceClassificationDP(args=args)
else:
    sc_modelling = SequenceClassification(args=args)

sc_modelling.train_model()
