import os

import pandas as pd

from local_constants import MODEL_DIR
import json
from utils.helpers import read_json


def all_metrics_to_csv():
    all_data = []
    for model_name in os.listdir(MODEL_DIR):
        metrics_file = os.path.join(MODEL_DIR, model_name, 'metrics', 'key_metrics.json')
        if os.path.isfile(metrics_file):
            data = read_json(metrics_file)


            if 'key_metrics' in data.keys():
                if len(data['key_metrics']) > 1:
                    data['key_metrics'][0]['loss'] = data['key_metrics'][2]['best_loss']['loss']
                    data['key_metrics'][0]['acc'] = data['key_metrics'][1]['best_acc']['acc']
                data = data['key_metrics'][0]

            if not data['dp']:
                data['epsilon'] = None
                data['delta'] = None
                data['lot_size'] = None
            if 'best_metrics' in data.keys():
                data['loss'] = data['best_metrics']['loss']['score']
                data['acc'] = data['best_metrics']['acc']['score']
                data['f_1'] = data['best_metrics']['f1']['score']

                data.pop('best_metrics', None)


                print()



            all_data.append(data)


    df = pd.DataFrame.from_records(all_data)

    df.to_csv('metadata/model_metrics.csv', sep='\t')

all_metrics_to_csv()

# def init_file():
#     for model_name in os.listdir(MODEL_DIR):
#         metrics_file = os.path.join(MODEL_DIR, model_name, 'metrics', 'key_metrics.json')
#         if os.path.isfile(metrics_file):
#             data = read_json(metrics_file)
#             key_names_1 = list(data['key_metrics'][0].keys())
#             key_names_1.remove('output_name')
#             key_names_1.remove('best_metrics')
#
#
#             key_names_2 = list(data['key_metrics'][0]['best_metrics'].keys())
#
#             header = 'model_name' + '\t' + '\t'.join(key_names_2) + '\t' + '\t'.join(key_names_1)
#
#             with open('metadata/model_metrics.csv', 'w', encoding='utf-8') as file:
#                 file.write(header)
#             break


