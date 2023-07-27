# to be created

from textwrap import wrap
from ner.modelling_utils.input_args import NERArgParser
from ner.modelling_utils.ner_modelling import NERModelling
from shared.modelling_utils.helpers import create_data_loader

# from ner.data_utils.get_dataset import tokenize_and_align_labels_for_dataset
from shared.data_utils.helpers import DatasetWrapper

sc_parser = NERArgParser()

args = sc_parser.parser.parse_args()

# args.model_name = 'babba'
# For NER models: these should be located in below directory
args.custom_model_dir = "ner/models"
args.evaluate_during_training = False
args.load_alvenir_pretrained = True
args.differential_privacy = False
args.test = True
args.eval_batch_size = 1
args.normalize_conf = "true"
# args.max_length = 512
# args.test_data = "bilou_val.jsonl"
# args.concat_bilu = True

modelling = NERModelling(args)

modelling.load_data(train=False, test=args.test)

# for i in range(len(modelling.data.test)):
# modelling.data.test[i]["ner_tags"] = [
# modelling.label2id[tag] for tag in modelling.data.test[i]["tags"]
#    ]


def tokenize_and_align_labels_for_dataset(dataset, tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=args.max_length,
        )

        labels = []
        labels_tokenized = []

        for i, label in enumerate(examples["tags"]):
            label = [modelling.label2id[lab] for lab in label]
            label_tokenized = tokenizer.tokenize(" ".join(examples["tokens"][i]))
            label_tokenized.insert(0, "-100")
            label_tokenized.append("-100")

            # Map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            labels_tokenized.append(label_tokenized)

        # ToDo: labels tokenized instead of labels?
        tokenized_inputs["labels"] = labels
        # tokenized_inputs["labels_tokenized"] = labels_tokenized

        return tokenized_inputs

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format("torch")

    #    tokenized_dataset_new = tokenized_dataset.remove_columns(
    #       ["ner_tags", "labels_tokenized", "tokens"]
    #  )

    return tokenized_dataset


tokenized = tokenize_and_align_labels_for_dataset(
    dataset=modelling.data.test, tokenizer=modelling.tokenizer
)

wrapped = DatasetWrapper(tokenized)


wrapped, test_loader = create_data_loader(
    # data_wrapped=modelling.tokenize_and_wrap_data(modelling.data.test),
    data_wrapped=wrapped,
    data_collator=modelling.data_collator,
    batch_size=modelling.args.eval_batch_size,
    shuffle=False,
)

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
