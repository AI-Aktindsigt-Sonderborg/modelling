from transformers import pipeline

import pandas as pd

ner = pipeline(task='ner',
               model='../ner/models/24-akt-mlm-BIO/best_model',
               aggregation_strategy='first')

sentence = 'Kasper Schjødt-Hansen er medejer i virksomheden Alvenir Aps og har ofte ekstrem hovedpine.' \
           ' Han bor på Blegdamsvej 74, 2100 København Ø. ' \
           'Hans tlf nummer er 12345560 og han er fra Danmark.'



result = ner(sentence)
print()
print("Input sætning:")
print(sentence)


pd_result = pd.DataFrame.from_records(result)
print()
print("Entiteter: ")
print(pd_result)

print()