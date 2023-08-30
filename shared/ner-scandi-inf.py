from transformers import pipeline

import pandas as pd

ner = pipeline(task='ner',
               model='../ner/models/24-akt-mlm-BIO/best_model',
               aggregation_strategy='first')

sentence = 'Kasper Schjødt-Hansen er medarbejder i virksomheden Alvenir Aps og har ofte ekstrem hovedpine.' \
           ' Han bor på Blegdamsvej 85, 2100 København Ø som ligger i københavns kommune.' \
           ' Hans tlf nummer er 12345560 og han er fra Danmark.'



result = ner(sentence)
print()
print("Input sætning:")
print(sentence)


pd_result = pd.DataFrame.from_records(result)
print()
print("Entiteter: ")
print(pd_result)

print()