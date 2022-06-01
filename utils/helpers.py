import sys
import time
from typing import Optional

import numpy as np
from opacus.validators import ModuleValidator

def validate_model(model):
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model is not compatible for DF with opacus. Please fix errors.")
        print(errors)
        sys.exit(0)
    else:
        print("Model is compatible for DP with opacus.")

def fix_and_validate(model):
    model = ModuleValidator.fix_and_validate(model)
    return model



class TimeCode:
    def __init__(self):
        self.start_time = time.time()

    def how_long_since_start(self, prefix: Optional[str] = None):
        time_end = time.time()
        final_time_seconds = round(time_end - self.start_time, 2)
        final_time_minutes = round(final_time_seconds / 60, 2)
        print_string = f"Time it took: {final_time_seconds} seconds, {final_time_minutes} minutes"
        if prefix:
            print_string = prefix + print_string
        print(print_string)

def accuracy(preds, labels):
    return (preds == labels).mean()

def evaluate(model, val_dataloader):
    model.eval()

    loss_arr = []
    accuracy_arr = []

    for batch in val_dataloader:
        # compute output
        output = model(input_ids=batch["input_ids"].to('cuda'),
                       attention_mask=batch["attention_mask"].to('cuda'),
                       labels=batch["labels"].to('cuda'))


        # batch = tuple(t.to('cuda') for t in batch)

        loss, logits = output[:2]

        preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        labels = batch["labels"].cpu().numpy()

        labels_flat = labels.flatten()
        preds_flat = preds.flatten()
        filtered = [[xv, yv] for xv, yv in zip(labels_flat, preds_flat) if xv != -100]

        labels_filtered = np.array([x[0] for x in filtered])
        preds_filtered = np.array([x[1] for x in filtered])
        loss_arr.append(loss.item())
        # accuracy_arr.append(accuracy(preds_filtered, labels_filtered))

    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr)

