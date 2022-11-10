import os
import pickle

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, DataCollatorForLanguageModeling, AutoTokenizer

from data_utils.helpers import DatasetWrapper
from modelling_utils.custom_modeling_bert import BertForMaskedLM, BertOnlyMLMHeadCustom
from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from modelling_utils.input_args import MLMArgParser

svm_filename = 'classifiers/svm_03.sav'

os.environ["WANDB_DISABLED"] = "true"
mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

args.eval_batch_size = 10
args.load_alvenir_pretrained = True
args.eval_data = 'train_classified.json'
args.model_name = 'last_model'
mlm_eval = MLMUnsupervisedModelling(args=args)
mlm_eval.load_data(train=False)
mlm_eval.model = BertForMaskedLM.from_pretrained(mlm_eval.local_alvenir_model_path)

tokenizer = AutoTokenizer.from_pretrained(mlm_eval.local_alvenir_model_path)

def tokenize_and_wrap_data(data: Dataset):
    def tokenize_function(examples):
        # Remove empty lines
        examples['text'] = [
            line for line in examples['text'] if len(line) > 0 and not line.isspace()
        ]

        return tokenizer(
            examples['text'],
            # padding='max_length',
            truncation=True,
            max_length=512,
            # We use this option because DataCollatorForLanguageModeling (see below)
            # is more efficient when it receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized = data.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=['text', 'label'],
        load_from_cache_file=False,
        desc="Running tokenizer on dataset line_by_line",
    )

    wrapped = DatasetWrapper(tokenized)

    return wrapped

def create_embeddings(data_loader: DataLoader, model):
    model.eval()
    model = model.to('cuda')

    # with tqdm(val_loader, unit="batch", desc="Batch") as batches:
    list_embed = []
    for batch in tqdm(data_loader, unit="batch", desc="Eval"):
        # for batch in val_loader:
        output = model(input_ids=batch["input_ids"].to(args.device),
                       attention_mask=batch["attention_mask"].to(args.device),
                       labels=batch["labels"].to(args.device),
                       output_hidden_states=True,
                       return_dict=True)
        embeddings = torch.mean(output.hidden_states[-2], dim=1).detach().cpu().numpy()

        # logits = output.logits.detach().cpu().numpy()
        # preds = np.argmax(logits, axis=-1)

        list_embed.append(embeddings)
    all_embeddings = np.concatenate(list_embed)
    return all_embeddings

if __name__ == '__main__':


    eval_data_wrapped = tokenize_and_wrap_data(data=mlm_eval.eval_data)

    collator = DataCollatorForLanguageModeling(tokenizer=mlm_eval.tokenizer, mlm=False)

    eval_loader = DataLoader(dataset=eval_data_wrapped, batch_size=1,
                             collate_fn=collator)

    embeddings = create_embeddings(data_loader=eval_loader, model=mlm_eval.model)

    np.save(arr=embeddings, file='data/embeddings.npy')


    # classifier = svm.SVC(kernel='rbf', C=1)
    # classifier.fit(X=X_train, y=y_train)

    # pickle.dump(classifier, open(model_filename, 'wb'))

    # classifier = pickle.load(open(svm_filename, 'rb'))

    # loaded_model = pickle.load(open(model_filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    # # print("score:" + result)
    #
    #
    #
    # predictions = loaded_model.predict(X_test)
    # print(predictions)



    # eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    #
    # read_embed = np.load('data/test_embeddings.npy')
    #
    # config = BertConfig.from_pretrained(mlm_eval.local_alvenir_model_path)
    # new_head = BertOnlyMLMHeadCustom(config)
    # new_head.load_state_dict(torch.load(mlm_eval.local_alvenir_model_path + '/head_weights.json'))
    #
    # lm_head = new_head.to(mlm_eval.args.device)
    # mlm_eval.model.cls = lm_head
    #
    # eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    # print(f'eval_loss: {eval_loss}')
    # print(f'eval_acc: {eval_accuracy}')
