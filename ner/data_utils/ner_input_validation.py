import sys

from shared.utils.helpers import read_json_lines
from ner.local_constants import DATA_DIR
import traceback

args = sys.argv[1:]

filename = args[0]
filename_bilou = args[1]

data = read_json_lines(DATA_DIR, filename)
data_bilou = read_json_lines(DATA_DIR, filename_bilou)
error_count = 0
annotation_errors = 0
annotation_counter = 0
j = 0
for i, obs in enumerate(data):
    try:
        assert isinstance(obs['id'], int)
        assert isinstance(obs['process_id'], str)
        assert isinstance(obs['display_name'], str)
        for k, pdf_text in enumerate(obs['pdf_text']):
            page_num = f'page_no:{k + 1}'
            assert isinstance(pdf_text, str)
            current_page_annotations = obs['text_annotation'][page_num]

            for j, annotation in enumerate(current_page_annotations):
                try:
                    assert isinstance(annotation, dict)
                    assert isinstance(annotation['annotation']['id'], int)
                    assert isinstance(annotation['annotation']['state'], str)
                    assert isinstance(annotation['annotation']['content'], str)
                    assert isinstance(annotation['annotation']['annotation'],
                                      str)
                    assert isinstance(annotation['annotation']['start'], int)
                    assert isinstance(annotation['annotation']['end'], int)
                    assert isinstance(annotation['annotation']['created'], str)
                    assert isinstance(
                        annotation['annotation']['category']['unsure'], bool)
                    assert isinstance(
                        annotation['annotation']['category']['guid'], str)
                    index = pdf_text[annotation['annotation']['start']:
                                     annotation['annotation']['end']]
                    content = annotation['annotation']['content']
                    assert index == content

                except:
                    print(
                        f"data med linjenummer {i + 1} med id {obs['id']} fejlede paa annotation nummer {j}.")
                    print(traceback.format_exc())
                    annotation_errors = annotation_errors + 1
            annotation_counter = annotation_counter + j

    except Exception as e:
        error_count = error_count + 1
        print(
            f"data med linjenummer {i + 1} med id {obs['id']} fejlede paa input validering.")
        print(f'page_num: {page_num}, annotation_num: {j}')
        print(traceback.format_exc())
        print(f'{e}')

bilou_error_count = 0
for i, obs in enumerate(data_bilou):
    if len(obs['words']) != len(obs['tags']):
        bilou_error_count = bilou_error_count + 1
        print(
            f'len(Words) != len(tags): Bilou data med linjenummer {i + 1} fejlede')
    if obs['faulty_annotation']:
        bilou_error_count = bilou_error_count + 1
        print(
            f'faulty_annotation fejl: Bilou data med linjenummer {i + 1} fejlede')

print(
    f'Antal dokumenter med generelle fejl: {error_count}, antal dokumenter i alt: {len(data)}')
print(
    f'Antal annotations med fejl: {annotation_errors} (annotations i alt: {annotation_counter})')
print(f'Antal bilou fejl: {bilou_error_count}, bilou i alt: {len(data_bilou)}')
