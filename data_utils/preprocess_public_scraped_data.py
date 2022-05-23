import json
import re
from typing import List
data = []

def preprocess_public_sborg(data_path: str = '../data/kommunal_data_test.json',
                            save_data: bool = True, languages: List[str] = ['da_DK']):


    with open(data_path, 'rb') as file:
        for index, line in enumerate(file):

            data_dict = json.loads(line)
            data_dict['id'] = index
            data.append(data_dict)

            title = data_dict['title'].split(' | ')[0]
            data_dict['new_title'] = title

            body = re.split(r'Opdateret \d\d.\d\d.\d\d\d\d', data_dict['text'])
            if len(body) == 2:
                data_dict['body'] = body[1]
            else:
                data_dict['body'] = ''

    for lang in languages:
        if lang == 'da_DK':
            # english_locale = [x for x in data if (x['og_locale'] == 'en_GB' or '//en.' in x['url'] or '/en/' in x['url'])]
            # german_locale = [x for x in data if x['og_locale'] == 'de_DE' or '//de.' in x['url'] or '/de/' in x['url']]
            # danish_locale = [x for x in data if x['og_locale'] == 'da_DK']
            # empty_locale = [x for x in data if (len(x['og_locale']) == 0 and not
            # ( '//en.' in x['url'] or '/en/' in x['url'] or
            #  '//de.' in x['url'] or '/de/' in x['url']))]

            all_danish = [x for x in data if x['og_locale'] == lang or (len(x['og_locale']) == 0 and not
            ( '//en.' in x['url'] or '/en/' in x['url'] or
             '//de.' in x['url'] or '/de/' in x['url']))]

            subset = [x for x in all_danish if len(x['body']) > 0]
            if save_data:

                # return subset

                with open(f"../data/{lang}_subset.json", "w") as outfile:
                    outfile.write(json.dumps(subset))

            # print()


# preprocess_public_sborg()

with open(f"../data/da_DK_subset.json", "rb") as file:
    data = json.load(file)

print()



