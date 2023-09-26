from typing import OrderedDict

from datasets import load_dataset


def get_wikiann_train(subset: int = None):
    if subset:
        dataset = load_dataset("wikiann", "da", split=f"train[:{subset}%]")
    else:
        dataset = load_dataset("wikiann", "da", split="train")
    return dataset


def get_dane_train(subset: int = None):
    if subset:
        dataset = load_dataset("dane", "da", split=f"train[:{subset}%]")
    else:
        dataset = load_dataset("dane", "da", split="train")
    return dataset


def get_dane_val(subset: int = None):
    if subset:
        dataset = load_dataset("dane", "da", split=f"validation[:{subset}%]")
    else:
        dataset = load_dataset("dane", "da", split="validation")
    return dataset


def get_dane_test(subset: int = None):
    if subset:
        dataset = load_dataset("dane", "da", split=f"test[:{subset}%]")
    else:
        dataset = load_dataset("dane", "da", split="test")
    return dataset


def get_wikiann_val(subset: int = None):
    if subset:
        dataset = load_dataset("wikiann", "da", split=f"validation[:{subset}%]")
    else:
        dataset = load_dataset("wikiann", "da", split="validation")
    return dataset


def tokenize_and_align_labels_for_dataset(dataset, tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=512,
        )

        labels = []
        labels_tokenized = []

        for i, label in enumerate(examples["ner_tags"]):
            # label = label2id[label]
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


def get_label_list_dane():
    label_list = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ]
    id2label = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    label2id = {v: k for k, v in id2label.items()}

    label2weight = OrderedDict()

    for i, label in enumerate([label2id[x] for x in label_list]):
        if label == 0:
            label2weight[label] = 1
        else:
            label2weight[label] = 2

    return label_list, id2label, label2id, label2weight
