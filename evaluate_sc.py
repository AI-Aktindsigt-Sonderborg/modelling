from torch.utils.data import DataLoader

from modelling_utils.input_args import SequenceModellingArgParser
from modelling_utils.sequence_classification import SequenceClassification

sc_parser = SequenceModellingArgParser()

label_dict = {'Beskæftigelse og integration': 0, 'Børn og unge': 1,
              'Erhverv og turisme': 2, 'Klima, teknik og miljø': 3,
              'Kultur og fritid': 4, 'Socialområdet': 5,
              'Sundhed og ældre': 6, 'Økonomi og administration': 7}

LABELS = list(label_dict)

args = sc_parser.parser.parse_args()

# args.model_name = 'sarnikowski/convbert-small-da-cased'
# args.model_name = 'last_model-2022-12-16_21-55-28'
args.labels = LABELS
args.evaluate_during_training = False
args.load_alvenir_pretrained = False
args.device = 'cpu'
args.test_data = 'test_local.json'

modelling = SequenceClassification(args)

modelling.load_data(train=False, test=True)

test_data_wrapped = modelling.tokenize_and_wrap_data(data=modelling.test_data)
test_loader = DataLoader(dataset=test_data_wrapped,
                         collate_fn=modelling.data_collator,
                         batch_size=modelling.args.eval_batch_size)

model = modelling.get_model

acc, f_1, loss = modelling.evaluate(
    model=model,
    val_loader=test_loader,
    conf_plot=True)

print(acc)
print(f_1)
print(loss)
