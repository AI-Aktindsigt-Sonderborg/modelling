import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

from local_constants import FILTERED_SCRAPE_DIR, DATA_DIR


def score_gpt2(sentence, model, tokenizer):
    words = sentence.split(" ")
    new_text = " ".join(words[0:200])
    tensor_input = tokenizer.encode(new_text, return_tensors='pt').to(device)

    if len(tensor_input[0]) > 1000:
        return 50000.0

    if max(tensor_input[0]) > 50257:
        # print("Bad sentence")
        # print(new_sentence)
        return 51000.0

    with torch.no_grad():
        result = model(tensor_input, labels=tensor_input).loss

    return np.exp(result.cpu().numpy())


device = "cuda"
with torch.no_grad():
    model_id = "pere/norwegian-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelWithLMHead.from_pretrained(model_id)
    model = model.to(device)
    model.eval()


def read_sentences_compute_ppl(in_file: str = 'unique_sentences.json',
                               out_file: str = 'unique_sentences_ppl.json'):
    approved_content = []
    disapproved_content = []
    criterias = []
    with open(os.path.join(FILTERED_SCRAPE_DIR, in_file), 'r', encoding='utf-8') as file, \
        open(os.path.join(DATA_DIR, 'data_testing/approved_sentences_ppl.txt'), 'w',
             encoding='utf-8') as approved_sentences, \
        open(os.path.join(DATA_DIR, 'data_testing/disapproved_sentences_ppl.txt'), 'w',
             encoding='utf-8') as disapproved_sentences:

        for i, line in enumerate(file):
            if i % 5000 == 0:
                print(i)
            data_dict = json.loads(line)

            # text = data_dict['text'].replace(u'\\.', '')
            # new = re.sub('\.', '', data_dict['text'])
            # new = re.sub('[\.|,]', ' ', data_dict['text'])
            # criteria_matching = text_processor.matches_all_criteria(data_dict['text'])
            # criterias.append(criteria_matching)
            # if not criteria_matching: #or not (data_dict['text'][0].isupper()):
            #     continue

            try:
                # greedy_score = score(data_dict['text'])
                ppl_score = score_gpt2(data_dict['text'], model, tokenizer)
                data_dict['ppl_score'] = ppl_score
            except Exception as e:
                print(e)
                print(data_dict['text'])
                sys.exit()

            if ppl_score < 1000.0:
                # approved_content.append(data_dict)
                approved_sentences.write(
                    f"{data_dict['kommune']} -- {data_dict['id']} -- {data_dict['sentence']} -- "
                    f"{data_dict['ppl_score']} -- {data_dict['text']}\n")
            else:
                # disapproved_content.append(data_dict)
                disapproved_sentences.write(
                    f"{data_dict['kommune']} -- {data_dict['id']} -- {data_dict['sentence']} -- "
                    f"{data_dict['ppl_score']} -- {data_dict['text']}\n")

    # return approved_content, disapproved_content

    # def compute_ppl(self, in_file: str = 'unique_sentences.json',
    #             out_file_name: str = 'unique_sentences_ppl.json'):
    #
    #
    #     model, tokenizer = self.load_model_for_ppl()
    #
    #     approved_counter = 0
    #     disapproved_counter = 0
    #     with open(os.path.join(FILTERED_SCRAPE_DIR, in_file), 'r', encoding='utf-8') as file, \
    #         open(os.path.join(DATA_DIR, f'data_testing/{out_file_name}'), 'w',
    #              encoding='utf-8') as out_file:
    #
    #         for i, line in enumerate(file):
    #             if i % 5000 == 0 and i != 0:
    #                 print(i)
    #             data_dict = json.loads(line)
    #
    #             # text = data_dict['text'].replace(u'\\.', '')
    #             # new = re.sub('\.', '', data_dict['text'])
    #             # new = re.sub('[\.|,]', ' ', data_dict['text'])
    #             # criteria_matching = text_processor.matches_all_criteria(data_dict['text'])
    #             # criterias.append(criteria_matching)
    #             # if not criteria_matching: #or not (data_dict['text'][0].isupper()):
    #             #     continue
    #
    #             try:
    #                 # greedy_score = score(data_dict['text'])
    #                 ppl_score = score_gpt2(data_dict['text'], model, tokenizer)
    #                 if ppl_score < self.ppl_threshold:
    #                     approved_counter += 1
    #                 else:
    #                     disapproved_counter += 1
    #
    #                 data_dict['ppl_score'] = str(ppl_score)
    #             except Exception as e:
    #                 print(e)
    #                 print(data_dict['text'])
    #                 sys.exit()
    #
    #             json.dump(data_dict, out_file)
    #             out_file.write('\n')
    #
    #             # if ppl_score < 1000.0:
    #             #     # approved_content.append(data_dict)
    #             #     approved_sentences.write(
    #             #         f"{data_dict['kommune']} -- {data_dict['id']} -- {data_dict['sentence']} -- "
    #             #         f"{data_dict['ppl_score']} -- {data_dict['text']}\n")
    #         print(f'ppl disapproved: {disapproved_counter}')
    #         print(f'ppl approved: {approved_counter}')
    #         print(f'total: {i+1}, pct approved: {float(approved_counter/(i+1))*100.0}%')


if __name__ == '__main__':
    from utils.helpers import TimeCode

    code_timer = TimeCode()
    read_sentences_compute_ppl()
    code_timer.how_long_since_start()
    # threshold_higher_dis = [x for x in disapproved_content if x['ppl_score'] > 250]
    # print()
