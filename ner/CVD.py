from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.data_utils.prep_data import NERDataPreprocessing

prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()
prep_args.bilou_input_file = 'bilou_entities_kasper_all'
prep_args.create_bilou = True
data_prep = NERDataPreprocessing(prep_args)

prep_args.entities = DataPrepConstants.standard_ner_entities

if prep_args.create_bilou:
    data_prep.create_bilou(args=prep_args)

bilou = data_prep.filter_entities(prep_args)

prep_args.entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED",
                             "ORGANISATION", "KOMMUNE", "TELEFONNUMMER",
                             "FORBRYDELSE", "CPR"]

if prep_args.create_bilou:
    data_prep.create_bilou(args=prep_args)

bilou = data_prep.filter_entities(prep_args)

# data_prep.train_val_test_to_json_split(
#     args=prep_args,
#     data=bilou,
#     train_size=prep_args.split,
#     test_size=prep_args.test_size,
#     train_outfile=prep_args.train_outfile,
#     val_outfile=prep_args.val_outfile,
#     test_outfile=prep_args.test_outfile,
#     add_dane=prep_args.add_dane,
# )

