import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from modelling_utils.custom_modeling_bert import BertForMaskedLM, BertOnlyMLMHeadCustom
from modelling_utils.mlm_modelling import MLMUnsupervisedModelling
from modelling_utils.input_args import MLMArgParser

os.environ["WANDB_DISABLED"] = "true"

if __name__ == '__main__':
    mlm_parser = MLMArgParser()
    args = mlm_parser.parser.parse_args()

    args.eval_batch_size = 10
    args.load_alvenir_pretrained = False
    args.eval_data = 'test_classified.json'
    mlm_eval = MLMUnsupervisedModelling(args=args)
    mlm_eval.load_data(train=False)
    mlm_eval.model = BertForMaskedLM.from_pretrained(args.model_name)




    eval_data_wrapped = mlm_eval.tokenize_and_wrap_data(data=mlm_eval.eval_data)
    eval_loader = DataLoader(dataset=eval_data_wrapped,
                             collate_fn=mlm_eval.data_collator,
                             batch_size=mlm_eval.args.eval_batch_size)



    model = mlm_eval.model.to('cuda')

    # with tqdm(val_loader, unit="batch", desc="Batch") as batches:
    all_logits = []
    for batch in tqdm(eval_loader, unit="batch", desc="Eval"):
        # for batch in val_loader:
        output = model(input_ids=batch["input_ids"].to(args.device),
                       attention_mask=batch["attention_mask"].to(args.device),
                       labels=batch["labels"].to(args.device),
                       output_hidden_states=True,
                       return_dict=True)

        logits = output.logits.detach().cpu().numpy()
        all_logits.append(logits)
        model.predictions()
    all_embeddings = np.concatenate(all_logits)


    eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)



    config = BertConfig.from_pretrained(mlm_eval.local_alvenir_model_path)
    new_head = BertOnlyMLMHeadCustom(config)
    new_head.load_state_dict(torch.load(mlm_eval.local_alvenir_model_path + '/head_weights.json'))

    lm_head = new_head.to(mlm_eval.args.device)
    mlm_eval.model.cls = lm_head

    eval_loss, eval_accuracy = mlm_eval.evaluate(mlm_eval.model, eval_loader)
    print(f'eval_loss: {eval_loss}')
    print(f'eval_acc: {eval_accuracy}')
