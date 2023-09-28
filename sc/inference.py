from sc.local_constants import MODEL_DIR
from sc.modelling_utils.input_args import SequenceModellingArgParser
from sc.modelling_utils.sequence_classification import SequenceClassification

sc_parser = SequenceModellingArgParser()

label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
              'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)

args = sc_parser.parser.parse_args()
args.labels = LABELS

args.load_alvenir_pretrained = True

args.model_name = 'ts1'
args.custom_model_dir = MODEL_DIR

modelling = SequenceClassification(args)

model = modelling.get_model()

prediction = modelling.predict(model=model,
                                sentence="Dette er en sætning om en børnehave",
                                label="Børn og unge")

print(f"Sætning: {prediction.sentence}")
print(f"Sand KL kategori: {prediction.label}")
print(f"Modellens forudsagte KL kategori: {prediction.prediction}")
