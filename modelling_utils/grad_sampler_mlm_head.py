from typing import Dict

import numpy as np
import torch
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample import register_grad_sampler
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertForMaskedLM
from modelling_utils.custom_modeling_bert import BertOnlyMLMHeadCustom, BertPredictionHeadTransform, BertLMPredictionHeadCustom
# from transformers.models.bert.modeling_bert import
from modelling_utils.custom_modeling_bert import BertLMPredictionHeadCustom

# @register_grad_sampler(BertLMPredictionHeadCustom)
# def compute_bert_lm_head_grad_sample(
#     layer: BertLMPredictionHeadCustom, activations: torch.Tensor, backprops: torch.Tensor
# ) -> Dict[nn.Parameter, torch.Tensor]:
#     """
#     Computes per sample gradients for ``nn.Linear`` layer
#     Args:
#         layer: Layer
#         activations: Activations
#         backprops: Backpropagations
#     """
#     print(activations)
#     print(backprops)
#     gs = torch.einsum("n...i,n...j->nij", backprops, activations)
#     ret = {layer.weight: gs}
#     if layer.bias is not None:
#         ret[layer.bias] = torch.einsum("n...k->nk", backprops)
#
#     return ret
from utils.helpers import validate_model


def compute_loss(prediction_scores, labels, config):
    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))

    return masked_lm_loss

def replace_bert_head(model_tag: str = "NbAiLab/nb-bert-base", device: str = 'cuda'):

    # model_tag = "NbAiLab/nb-bert-base"
    model = BertForMaskedLM.from_pretrained(model_tag)

    config = BertConfig.from_pretrained(model_tag)
    lm_head = BertOnlyMLMHeadCustom(config)
    lm_head = lm_head.to(device)

    # print(f"Before wrapping: {lm_head}")
    # gs_lin_mod = GradSampleModule(lm_head)
    # print(f"After wrapping : {gs_lin_mod}")
    #
    # sequence_output = np.load("sequence_output.npy")[0:2, :, :]
    # labels = np.load("labels.npy")[0:2, :]
    # labels = torch.Tensor(labels).to(torch.int64).to(device)
    # sequence_tensor = torch.Tensor(sequence_output).to(device)
    # prediction_scores = gs_lin_mod.forward(sequence_tensor)
    # loss = compute_loss(prediction_scores, labels, config)
    #
    # loss.backward()


    # old_bias = model.cls.predictions.bias
    #
    # old_decoder_weights = model.cls.predictions.decoder.weight
    # old_dense_weights = model.cls.predictions.transform.dense.weight
    # old_layer_norm_weights = model.cls.predictions.transform.LayerNorm.weight
    #
    # old_decoder_bias = model.cls.predictions.decoder.bias
    # old_dense_bias = model.cls.predictions.transform.dense.bias
    # old_layer_norm_bias = model.cls.predictions.transform.LayerNorm.bias


    model.cls = lm_head
    # model.cls.predictions = lm_head

    # model.cls.predictions.bias = old_bias
    #
    # model.cls.predictions.decoder.weight = old_decoder_weights
    # model.cls.predictions.transform.dense.weight = old_dense_weights
    # model.cls.predictions.transform.LayerNorm.weight = old_layer_norm_weights
    #
    # model.cls.predictions.decoder.bias = old_decoder_bias
    # model.cls.predictions.transform.dense.bias = old_dense_bias
    # model.cls.predictions.transform.LayerNorm.bias = old_layer_norm_bias


    return model

if __name__ == '__main__':
    model_tag = "NbAiLab/nb-bert-base"
    config = BertConfig.from_pretrained(model_tag)
    device = 'cuda'
    #
    model = replace_bert_head()

    model = model.train()

    # model = GradSampleModule(model)

    sequence_output = np.load("sequence_output.npy")[0:2, :, :]
    labels = np.load("labels.npy")[0:2, :]
    labels = torch.Tensor(labels).to(torch.int64).to(device)
    sequence_tensor = torch.Tensor(sequence_output).to(device)
    prediction_scores = model.cls.forward(sequence_tensor)
    loss = compute_loss(prediction_scores, labels, config)

    loss.backward()
    #
    #
    #
    # validate_model(model)
    # device = "cuda"



    model_tag = "NbAiLab/nb-bert-base"
    config = BertConfig.from_pretrained(model_tag)
    lm_head = BertOnlyMLMHeadCustom(config)
    lm_head = lm_head.to(device)





    print(f"Before wrapping: {lm_head}")
    gs_lin_mod = GradSampleModule(lm_head)
    print(f"After wrapping : {gs_lin_mod}")

    sequence_output = np.load("sequence_output.npy")[0:2, :, :]
    labels = np.load("labels.npy")[0:2, :]
    labels = torch.Tensor(labels).to(torch.int64).to(device)
    sequence_tensor = torch.Tensor(sequence_output).to(device)
    prediction_scores = gs_lin_mod.forward(sequence_tensor)
    loss = compute_loss(prediction_scores, labels, config)

    loss.backward()

    NotImplementedError: [NotImplementedError(
        "Model contains a trainable layer that Opacus doesn't currently support(cls.predictions:BertLMPredictionHeadCustom(\n  "
        "(transform): BertPredictionHeadTransform(\n    (dense): Linear(in_features=768, out_features=768, bias=True)\n    "
        "(transform_act_fn): GELUActivation()\n    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)\n  )\n "
        " (decoder): Linear(in_features=768, out_features=119547, bias=True)\n)). "
        "Please implement and register grad sampler for this layer. (See opacus.grad_sample.utils.register_grad_sampler)")]
