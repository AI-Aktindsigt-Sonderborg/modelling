"""
Script to evaluate specific NER model. Model folder should be located in
"../modelling/ner/models" and the model itself should be placed in
"..<model_name>/best_model" folder.

The data file used for evaluation should be placed in "../ner/data/preprocessed_data/"

For CLI arguments run python -m ´ner.eval_model -h´

Example call for model **excluding** CPR and FORBRYDELSE:
    ´python -m ner.eval_model --test_data vejen_PLAHOKT.jsonl
    --model_name ner-SAS-dp --concat_bilou true´

Example call for model **including** CPR and FORBRYDELSE:
    ´python -m ner.eval_model --test_data vejen_PLAHOKT.jsonl
    --model_name ner-SAS-FC --concat_bilou true´ --entities PLAHOKTFC

"""

from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader

sc_parser = NERArgParser()

args = sc_parser.parser.parse_args()

# Important that log_wandb is False
args.log_wandb = False

# For NER models: these should be located in below directory
args.custom_model_dir = "ner/models"
args.data_format = "bio"
args.load_alvenir_pretrained = True
args.replace_head = False
args.differential_privacy = False
args.test = True
args.eval_batch_size = 1
args.normalize_conf = "true"
args.max_length = 512

if args.entities == ["PLAHOKTFC"]:
    args.entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED",
     "ORGANISATION", "KOMMUNE", "TELEFONNUMMER",
     "FORBRYDELSE", "CPR"]

# args.entities = DataPrepConstants.standard_ner_entities

modelling = NERModelling(args)
modelling.load_data(train=False, test=args.test)

wrapped, test_loader = create_data_loader(
    data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.test),
    # data_wrapped=wrapped,
    data_collator=modelling.data_collator,
    batch_size=modelling.args.eval_batch_size,
    shuffle=False,
)

model = modelling.get_model()
model.config.label2id = modelling.label2id
model.config.id2label = modelling.id2label

eval_scores = modelling.evaluate(model=model, val_loader=test_loader,
                                 conf_plot=True)

print(eval_scores)
