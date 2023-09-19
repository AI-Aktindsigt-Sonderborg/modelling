from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.data_utils.data_prep_input_args import DataPrepArgParser
from ner.data_utils.prep_data import NERDataPreprocessing


if __name__ == "__main__":

    arg_parser = DataPrepArgParser()
    args = arg_parser.parser.parse_args()

    # Generel fixed args
    args.create_bilou = True
    args.create_bio_file = True
    # args.origin_input_file = "origin_vejen"

    # Create bilou
    args.bilou_input_file = "bilou_vejen"
    if args.create_bilou:
        NERDataPreprocessing.create_bilou(args=args)


    # Create data for model excluding CPR and Forbrydelse
    args.entities = DataPrepConstants.standard_ner_entities

    bilou = NERDataPreprocessing.filter_entities(args)
    # ---------------------------------------------------------

    # Create data for model including CPR and FORBRYDELSE
    args.entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED",
                                 "ORGANISATION", "KOMMUNE", "TELEFONNUMMER",
                                 "FORBRYDELSE", "CPR"]

    bilou = NERDataPreprocessing.filter_entities(args)
