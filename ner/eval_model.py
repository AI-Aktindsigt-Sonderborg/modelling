from ner.data_utils.custom_dataclasses import DataPrepConstants
from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader

# from ner.data_utils.get_dataset import tokenize_and_align_labels_for_dataset

sc_parser = NERArgParser()

args = sc_parser.parser.parse_args()
# Important that log_wandb is False
args.log_wandb = False

# args.model_name = 'babba'
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

args.entities = DataPrepConstants.standard_ner_entities

# args.test_data = "bilou_val.jsonl"
# args.concat_bilu = True

# data = read_json_lines(PREP_DATA_DIR, "bio_test_1508")

# for obs in data:
#   entity = obs["entities"][0]
#   obs["tags"] = [tag if tag[2:] == entity else "O" for tag in obs["tags"]]

# write_json_lines(out_dir=PREP_DATA_DIR, filename="single_entity_test", data=data)


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
