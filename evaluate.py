import argparse
import os

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, \
    AutoConfig, BertConfig, set_seed

from local_constants import DATA_DIR, MODEL_DIR
from modelling_utils.custom_modeling_bert import BertForMaskedLM, BertOnlyMLMHeadCustom, BertModel
from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

os.environ["WANDB_DISABLED"] = "true"


if __name__ == '__main__':
    mlm_parser = MLMArgParser()
    args = mlm_parser.parser.parse_args()
    # args.eval_data = 'val_new_scrape2.json'
    args.eval_data = 'val_scrape.json'
    args.model_name = 'NbAiLab_nb-bert-base-2022-08-11_14-28-23'
    args.model_name = os.path.join(MODEL_DIR, args.model_name, 'best_model')

    # args.eval_batch_size = 2
    # args.max_length = 32
    mlm_eval = MLMUnsupervisedModelling(args=args)
    mlm_eval.load_data(train=False)
    mlm_eval.model = BertForMaskedLM.from_pretrained(mlm_eval.args.model_name)
    # model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'NbAiLab/nb-bert-base')


    eval_data_wrapped = mlm_eval.tokenize_and_wrap_data(data=mlm_eval.eval_data)
    eval_loader = DataLoader(dataset=eval_data_wrapped,
                             collate_fn=mlm_eval.data_collator,
                             batch_size=mlm_eval.args.eval_batch_size)



    # eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    # print(f'eval_loss: {eval_loss}')
    # print(f'eval_acc: {eval_accuracy}')

    # mlm_eval.save_model(model=mlm_eval.model, data_collator=mlm_eval.data_collator,
    #                     output_dir="models/test/",
    #                     tokenizer=mlm_eval.tokenizer)
    #


    config = BertConfig.from_pretrained(mlm_eval.args.model_name)
    new_head = BertOnlyMLMHeadCustom(config)
    new_head.load_state_dict(torch.load(args.model_name + '/head_weights.json'))

    lm_head = new_head.to(mlm_eval.args.device)
    mlm_eval.model.cls = lm_head

    # config = BertConfig.from_pretrained(args.model_name)
    # mlm_eval.model = BertForMaskedLM(config=config)
    #
    # # bert_model = BertModel(config, add_pooling_layer=False)
    # from modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom
    #
    # lm_head = BertOnlyMLMHeadCustom(config)
    # lm_head = lm_head.to(args.device)
    # mlm_eval.cls = lm_head
    # mlm_eval.cls.load_state_dict(torch.load(args.model_name + '/head_weights.json'))




    eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    print(f'eval_loss: {eval_loss}')
    print(f'eval_acc: {eval_accuracy}')
    print()
