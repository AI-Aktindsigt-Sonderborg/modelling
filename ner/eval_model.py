# to be created

from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader

sc_parser = NERArgParser()

args = sc_parser.parser.parse_args()

# args.model_name = 'babba'
args.custom_model_dir = "ner/models"
args.evaluate_during_training = False
args.load_alvenir_pretrained = True
args.differential_privacy = False
args.test = True
args.test_data = "bilou_val.jsonl"
# args.concat_bilu = True

modelling = NERModelling(args)

modelling.load_data(train=False, test=args.test)

wrapped, test_loader = create_data_loader(
    data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.test),
    data_collator=modelling.data_collator,
    batch_size=modelling.args.eval_batch_size,
    shuffle=False
)

model = modelling.get_model()
model.config.label2id = modelling.label2id

eval_scores = modelling.evaluate(
    model=model,
    val_loader=test_loader,
    conf_plot=True)



print(eval_scores)


print(model.config.label2id)
