import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, DataCollatorForLanguageModeling, AutoTokenizer

from data_utils.helpers import DatasetWrapper
from local_constants import PREP_DATA_DIR
from modelling_utils.custom_modeling_bert import BertForMaskedLM, BertOnlyMLMHeadCustom
from modelling_utils.input_args import MLMArgParser
from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.helpers import read_jsonlines

svm_filename = 'classifiers/svm_alvenir.sav'

os.environ["WANDB_DISABLED"] = "true"
mlm_parser = MLMArgParser()
args = mlm_parser.parser.parse_args()

args.eval_batch_size = 10
args.load_alvenir_pretrained = True
args.eval_data = 'test_classified.json'
args.model_name = 'last_model'
mlm_eval = MLMUnsupervisedModelling(args=args)
mlm_eval.load_data(train=False)
# mlm_eval.model = BertForMaskedLM.from_pretrained(args.model_name)
mlm_eval.model = BertForMaskedLM.from_pretrained(mlm_eval.local_alvenir_model_path)

config = BertConfig.from_pretrained(mlm_eval.local_alvenir_model_path)
new_head = BertOnlyMLMHeadCustom(config)
new_head.load_state_dict(torch.load(mlm_eval.local_alvenir_model_path + '/head_weights.json'))

LM_HEAD = new_head.to(mlm_eval.args.device)
mlm_eval.model.cls = LM_HEAD

tokenizer = AutoTokenizer.from_pretrained(mlm_eval.local_alvenir_model_path)



# handle labels
label2id = {'Beskæftigelse og integration': 0, 'Børn og unge': 1, 'Erhvervsudvikling': 2,
            'Klima, teknik og miljø': 3, 'Kultur og fritid': 4, 'Socialområdet': 5,
            'Sundhed og ældre': 6, 'Økonomi og administration': 7, 'Økonomi og budget': 8}
label_list = list(label2id)
id2label = {v: k for k, v in label2id.items()}

# tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
def calc_f1_score(y_list, prediction_list, labels, conf_plot: bool = False):


    if conf_plot:
        y_list = [id2label[x] for x in y_list]
        prediction_list = [id2label[x] for x in prediction_list]
        conf_matrix = confusion_matrix(y_list, prediction_list, labels=labels,
                                       normalize='true')
        df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        # sn.set(font_scale=0.8)
        plt.figure(figsize=(10, 7))
        plot_labels = ['Besk. og int.', 'Børn/unge', 'Erhverv', 'Klima/tek/miljø','Kultur/fritid',
                       'Social', 'Sundhed/ældre', 'Øko./adm', 'Øko./budget']
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt='g', xticklabels=plot_labels,
                   yticklabels=plot_labels)
        # plt.savefig('plots/svm_03_confplot.png')
        plt.show()

    return precision_recall_fscore_support(y_list, prediction_list, labels=labels, average='micro'), \
           f1_score(y_true=y_list, y_pred=prediction_list, labels=labels, average=None)
if __name__ == '__main__':


    test_json = read_jsonlines(input_dir=PREP_DATA_DIR, filename='test_classified')

    test_sentences = [x['text'] for x in test_json]
    test_labels = [x['label'] for x in test_json]

    y_test = [label2id[x] for x in test_labels]


    eval_data_wrapped = tokenize_and_wrap_data(data=mlm_eval.eval_data)

    collator = DataCollatorForLanguageModeling(tokenizer=mlm_eval.tokenizer, mlm=False)

    eval_loader = DataLoader(dataset=eval_data_wrapped, batch_size=1,
                             collate_fn=collator)

    X_test = create_embeddings(data_loader=eval_loader, model=mlm_eval.model)

    classifier = pickle.load(open(svm_filename, 'rb'))

    result = classifier.score(X_test, y_test)
    # # print("score:" + result)
    #
    #
    #
    predictions = classifier.predict(X_test)
    # print(predictions)

    f1, f12 = calc_f1_score(y_list=y_test, prediction_list=predictions, labels=label_list,
                            conf_plot=True)

    print()

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
