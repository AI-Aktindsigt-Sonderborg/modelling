import argparse
import os

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForWholeWordMask, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, MODEL_FOR_MASKED_LM_MAPPING, \
    AutoConfig, BertConfig

from local_constants import DATA_DIR, MODEL_DIR
from modelling_utils.custom_modeling_bert import BertForMaskedLM, BertOnlyMLMHeadCustom
from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from utils.input_args import MLMArgParser

os.environ["WANDB_DISABLED"] = "true"


if __name__ == '__main__':
    mlm_parser = MLMArgParser()
    args = mlm_parser.parser.parse_args()
    # args.eval_data = 'val_new_scrape2.json'
    args.eval_data = 'val_1000.json'
    args.model_name = 'Geotrend_distilbert-base-da-cased-2022-08-11_12-04-48'
    args.model_name = os.path.join(MODEL_DIR, args.model_name, 'best_model')

    # args.eval_batch_size = 2
    # args.max_length = 32
    mlm_eval = MLMUnsupervisedModelling(args=args)
    mlm_eval.load_data(train=False)
    mlm_eval.model = BertForMaskedLM.from_pretrained(mlm_eval.args.model_name)

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
    # config = AutoConfig.from_pretrained('test/')
    # mlm_eval.model = BertForMaskedLM.from_pretrained("models/test/")

    # lm_head = model.cls
    # lm_head = lm_head.to(self.args.device)
    # lm_head = model.cls
    # config = BertConfig.from_pretrained(mlm_eval.args.model_name)
    # new_head = BertOnlyMLMHeadCustom(config)
    # new_head.load_state_dict(torch.load(args.model_name + '/model_weights.json'))
    #
    # lm_head = new_head.to(mlm_eval.args.device)
    # test_model.cls = lm_head

    mlm_eval.model.load_state_dict(torch.load(args.model_name + '/model_weights.json'))

    eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    print(f'eval_loss: {eval_loss}')
    print(f'eval_acc: {eval_accuracy}')
    print()
