from typing import Dict

import numpy as np
import torch
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample import register_grad_sampler
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLMPredictionHead


@register_grad_sampler(BertLMPredictionHead)
def compute_bert_lm_head_grad_sample(
    layer: BertLMPredictionHead, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    print(activations)
    print(backprops)
    gs = torch.einsum("n...i,n...j->nij", backprops, activations)
    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)

    return ret


def compute_loss(prediction_scores, labels):
    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, config.vocab_size), labels.view(-1))

    return masked_lm_loss

device = "cuda"
model_tag = "NbAiLab/nb-bert-base"
config = BertConfig.from_pretrained(model_tag)
lm_head = BertOnlyMLMHead(config)
lm_head = lm_head.to(device)

print(f"Before wrapping: {lm_head}")

gs_lin_mod = GradSampleModule(lm_head)
print(f"After wrapping : {gs_lin_mod}")

sequence_output = np.load("sequence_output.npy")[0:2, :, :]
labels = np.load("labels.npy")[0:2, :]
labels = torch.Tensor(labels).to(torch.int64).to(device)
sequence_tensor = torch.Tensor(sequence_output).to(device)
prediction_scores = lm_head.forward(sequence_tensor)
loss = compute_loss(prediction_scores, labels)

loss.backward()

# model = AutoModelForMaskedLM.from_pretrained(model_tag)


# validate_model(model)
