from transformers import pipeline

import pandas as pd

ner = pipeline(task='ner',
               model='../ner/models/24-akt-mlm-BIO/best_model',
               aggregation_strategy='first')

result = ner('Kasper Schjødt-Hansen er medejer i virksomheden Alvenir Aps og har ofte hovedpine.')

pd_result = pd.DataFrame.from_records(result)

print(pd_result)

print()