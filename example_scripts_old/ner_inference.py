"""
Inferens på NER model
===============================

Example script to predict NER categories from an input sentence.
Script should be run as module from project root folder "modelling" e.g.
- python -m example_scripts.ner_inference
"""
from ner.local_constants import MODEL_DIR
from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling

ner_parser = NERArgParser()

args = ner_parser.parser.parse_args()
args.entities = ["PERSON", "LOKATION", "ADRESSE", "HELBRED", "ORGANISATION",
                 "KOMMUNE", "TELEFONNUMMER"]


args.load_alvenir_pretrained = True

args.model_name = '<model_name>'
args.custom_model_dir = MODEL_DIR

modelling = NERModelling(args)

model = modelling.get_model()

prediction = modelling.predict(model=model,
                               sentence="Dette er en sætning om en børnehave "
                                        "i Sønderborg Kommune",
                               labels=None)

print(f"Sætning: {prediction.sentence}")
print(f"Sand KL kategori: {prediction.labels}")
print(f"Modellens forudsagte NER entiteter: {prediction.predictions}")
