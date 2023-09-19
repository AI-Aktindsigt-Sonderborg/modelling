from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.data_utils.prep_data import NERDataPreprocessing


if __name__ == "__main__":

    arg_parser = DataPrepArgParser()
    args = arg_parser.parser.parse_args()

    args.create_bilou = True
    args.origin_input_file = "origin_vejen"

    # Create data for model without CPR and Forbrydelse
    args.bilou_input_file = "bilou_vejen1"

    # data_prep = NERDataPreprocessing(args)

    args.entities = DataPrepConstants.standard_ner_entities

    if args.create_bilou:
        NERDataPreprocessing.create_bilou(args=args)

    bilou = NERDataPreprocessing.filter_entities(args)
    # ---------------------------------------------------------

    # Create data for model including CPR and FORBRYDELSE
    args.entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED",
                                 "ORGANISATION", "KOMMUNE", "TELEFONNUMMER",
                                 "FORBRYDELSE", "CPR"]

    args.bilou_input_file = "bilou_vejen2"

    if args.create_bilou:
        NERDataPreprocessing.create_bilou(args=args)

    bilou = NERDataPreprocessing.filter_entities(args)

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
