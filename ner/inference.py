from transformers import pipeline
import pandas as pd

ner = pipeline(task='ner',
               model='ner/models/sas-ner/best_model',
           aggregation_strategy='first')

sentence = 'Kasper Schjødt-Hansen er medarbejder i virksomheden Alvenir Aps og har ofte ekstrem hovedpine. Han bor på Blegdamsvej 85, 2100 København Ø som ligger i Københavns Kommune.' \
   ' Hans tlf nummer er 12345560 og han er fra Danmark. Blegamsvej er tæt på Fælledparken.'


result = ner(sentence)
print(pd.DataFrame.from_records(result))