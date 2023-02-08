import os

import pandas as pd

from ner.local_constants import MODEL_DIR
from shared.utils.helpers import read_json


def all_metrics_to_csv():
    all_data = []
    for model_name in os.listdir(MODEL_DIR):
        metrics_file = os.path.join(MODEL_DIR,
                                    model_name, 'metrics', 'key_metrics.json')

        if os.path.isfile(metrics_file):
            data = read_json(metrics_file)

            if 'key_metrics' in data.keys():
                if len(data['key_metrics']) > 1:
                    data['key_metrics'][0]['loss'] = \
                        data['key_metrics'][2]['best_loss']['loss']
                    data['key_metrics'][0]['acc'] = \
                        data['key_metrics'][1]['best_acc']['acc']
                data = data['key_metrics'][0]

            if not data['dp']:
                data['epsilon'] = None
                data['delta'] = None
                data['lot_size'] = None
            if 'best_metrics' in data.keys():
                data['loss'] = data['best_metrics']['loss']['score']
                if 'acc' in data['best_metrics'].keys():
                    data['accuracy'] = data['best_metrics']['acc']['score']
                else:
                    data['accuracy'] = data['best_metrics']['accuracy']['score']

                if 'f1' in data['best_metrics'].keys():
                    data['f_1'] = data['best_metrics']['f1']['score']
                else:
                    data['f_1'] = data['best_metrics']['f_1']['score']
            else:
                data['accuracy'] = data['acc']
                data.pop('acc', None)

            data.pop('best_metrics', None)

            all_data.append(data)

    df = pd.DataFrame.from_records(all_data)

    df.to_csv('metadata/model_metrics.csv', sep='\t')
    df.to_excel('metadata/model_metrics.xlsx', index=False)


all_metrics_to_csv()
