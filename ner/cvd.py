"""
Script to generate 'model-ready' test data from raw vejen data

origin file must be placed in folder '../modelling/ner/data'

Origin file must have '.jsonl' extension

Navigate to project root folder and run script via terminal:
    python -m ner.CVD --origin_input_file <origin_file_name>

Example run with origin file name 'origin_vejen.jsonl':
    python -m ner.CVD --origin_input_file origin_vejen

Important files created from script placed in '../modelling/ner/data':
    bio_PLAHOKT.jsonl --> file to test model excluding CPR and Forbrydelse
    bio_PLAHOKTFC.jsonl --> file to test model including CPR and Forbrydelse

"""


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
