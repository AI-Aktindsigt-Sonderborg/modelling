# to be created

from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader

sc_parser = NERArgParser()

args = sc_parser.parser.parse_args()

# args.model_name = 'last_model-2022-12-21_10-53-25'
args.evaluate_during_training = False
args.load_alvenir_pretrained = True
args.differential_privacy = False

modelling = NERModelling(args)

modelling.load_data(train=False, test=True)

wrapped, test_loader = create_data_loader(
    data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.test),
    data_collator=modelling.data_collator,
    batch_size=modelling.args.eval_batch_size,
    shuffle=False
)

model = modelling.get_model()

eval_scores = modelling.evaluate(
    model=model,
    val_loader=test_loader,
    conf_plot=True)

print(eval_scores)
