import os

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig

from modelling_utils.custom_modeling_bert import BertForMaskedLM, BertOnlyMLMHeadCustom
from modelling_utils.mlm_modelling import MLMModelling
from modelling_utils.input_args import MLMArgParser

os.environ["WANDB_DISABLED"] = "true"

if __name__ == '__main__':
    mlm_parser = MLMArgParser()
    args = mlm_parser.parser.parse_args()

    mlm_eval = MLMModelling(args=args)
    mlm_eval.load_data(train=False)
    mlm_eval.model = BertForMaskedLM.from_pretrained(mlm_eval.local_alvenir_model_path)

    eval_data_wrapped = mlm_eval.tokenize_and_wrap_data(data=mlm_eval.eval_data)
    eval_loader = DataLoader(dataset=eval_data_wrapped,
                             collate_fn=mlm_eval.data_collator,
                             batch_size=mlm_eval.args.eval_batch_size)

    config = BertConfig.from_pretrained(mlm_eval.local_alvenir_model_path)
    new_head = BertOnlyMLMHeadCustom(config)
    new_head.load_state_dict(torch.load(mlm_eval.local_alvenir_model_path + '/head_weights.json'))

    lm_head = new_head.to(mlm_eval.args.device)
    mlm_eval.model.cls = lm_head

    eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    print(f'eval_loss: {eval_loss}')
    print(f'eval_acc: {eval_accuracy}')
