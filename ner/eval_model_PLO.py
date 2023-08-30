# to be created

from textwrap import wrap
from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader

# from ner.data_utils.get_dataset import tokenize_and_align_labels_for_dataset
from shared.data_utils.helpers import DatasetWrapper

sc_parser = NERArgParser()

args = sc_parser.parser.parse_args()
# FixMe: do base evaluation with scandiner vs Alvenir
args.model_name = "saattrupdan/nbailab-base-ner-scandi"
# For NER models: these should be located in below directory
args.data_format = "bio"
args.load_alvenir_pretrained = False
args.evaluate_during_training = False
args.replace_head = False
args.differential_privacy = False
args.test = True
args.eval_batch_size = 4
args.normalize_conf = "true"
args.max_length = 512
# args.test_data = "bio_test1.jsonl"
args.entities = ["PERSON", "LOKATION", "ORGANISATION", "MISC"]
# args.concat_bilu = True

modelling = NERModelling(args)


# SE_acc = sqrt(p_class * ((1-p_class)/N)))
# np.sqrt(0.7627 * ((1-0.7627)/3742))


modelling.load_data(train=False, test=args.test)
# for i in range(len(modelling.data.test)):
#     modelling.data.test[i]['tags'] = [x if x in modelling.args.labels else "O" for x in modelling.data.test[i]['tags']]

modelling.id2label = {
    0: "B-LOKATION",
    1: "I-LOKATION",
    2: "B-ORGANISATION",
    3: "I-ORGANISATION",
    4: "B-PERSON",
    5: "I-PERSON",
    6: "B-MISC",
    7: "I-MISC",
    8: "O",
}
modelling.label2id = {
    "B-LOKATION": 0,
    "B-MISC": 6,
    "B-ORGANISATION": 2,
    "B-PERSON": 4,
    "I-LOKATION": 1,
    "I-MISC": 7,
    "I-ORGANISATION": 3,
    "I-PERSON": 5,
    "O": 8,
}


print()

wrapped, test_loader = create_data_loader(
    data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.test),
    # data_wrapped=wrapped,
    data_collator=modelling.data_collator,
    batch_size=modelling.args.eval_batch_size,
    shuffle=False,
)

modelling.id2label = {
    0: "B-LOKATION",
    1: "I-LOKATION",
    2: "B-ORGANISATION",
    3: "I-ORGANISATION",
    4: "B-PERSON",
    5: "I-PERSON",
    6: "B-MISC",
    7: "I-MISC",
    8: "O",
}
modelling.label2id = {
    "B-LOKATION": 0,
    "B-MISC": 6,
    "B-ORGANISATION": 2,
    "B-PERSON": 4,
    "I-LOKATION": 1,
    "I-MISC": 7,
    "I-ORGANISATION": 3,
    "I-PERSON": 5,
    "O": 8,
}


# modelling.id2label = {
#   "0": "B-LOKATION",
#  "1": "I-LOKATION",
# "2": "B-ORGANISATION",
# "3": "I-ORGANISATION",
# "4": "B-PERSON",
# "5": "I-PERSON",
# "6": "O"
# }

# modelling.label2id = {
#   "B-LOKATION": 0,
#   "B-ORGANISATION": 2,
#  "B-PERSON": 4,
# "I-LOKATION": 1,
# "I-ORGANISATION": 3,
# "I-PERSON": 5,
# "O": 6
# }

model = modelling.get_model()
model.config.label2id = modelling.label2id
model.config.id2label = modelling.id2label

eval_scores = modelling.evaluate(model=model, val_loader=test_loader, conf_plot=True)

print(eval_scores)
print("len test_loader")
print(len(test_loader))

print("len dataset")
print(len(modelling.data.test))

print(model.config.label2id)

print(model.config.id2label)
