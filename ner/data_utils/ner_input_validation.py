from shared.utils.helpers import read_json_lines
from ner.local_constants import DATA_DIR
import traceback

filename = 'user_annotated_output'
filename_bilou = "bilou_output_user_annotated"

data = read_json_lines(DATA_DIR, filename)
data_bilou = read_json_lines(DATA_DIR, filename_bilou)
error_count = 0
annotation_errors = 0
annotation_counter = 0
for i, obs in enumerate(data):
    try:
        assert isinstance(obs['id'], int)
        assert isinstance(obs['process_id'], str)
        assert isinstance(obs['display_name'], str)
        #assert isinstance(obs['file_extension'], str)
        for page in obs['pdf_text']:
            assert isinstance(page, str)
        k = 0
        for page_num in list(obs['text_annotation']):
            for j, annotation in enumerate(obs['text_annotation'][page_num]):
                try:
                    assert isinstance(annotation, dict)
                    assert isinstance(annotation['annotation']['id'], int)
                    assert isinstance(annotation['annotation']['state'], str)
                    assert isinstance(annotation['annotation']['content'], str)
                    assert isinstance(annotation['annotation']['annotation'], str)
                    assert isinstance(annotation['annotation']['start'], int)
                    assert isinstance(annotation['annotation']['end'], int)
                    assert isinstance(annotation['annotation']['created'], str)
                    assert isinstance(annotation['annotation']['category']['unsure'], bool)
                    assert isinstance(annotation['annotation']['category']['guid'], str)
                    index = obs['pdf_text'][k][annotation['annotation']['start']:annotation['annotation']['end']]
                    content = annotation['annotation']['content']
                    assert index == content
                except:
                    annotation_errors = annotation_errors + 1
            annotation_counter = annotation_counter + j
            k = k + 1
    except Exception as e:
        error_count = error_count + 1
        print(f"data nummer {i} med id {obs['id']} fejlede paa input validering.")
        print(f'page_num: {page_num}, annotation_num: {j}')
        print(traceback.format_exc())
        print(f'{e}')


bilou_error_count = 0
for i, obs in enumerate(data_bilou):
    try:
        assert len(obs['words']) == len(obs['tags'])
    except Exception as e:
        bilou_error_count = bilou_error_count + 1
        print(f'Bilou data med nummer {i} fejlede')
        print(traceback.format_exc())
        print(f'{e}')

print(f'Antal dokumenter med fejl: {error_count}, Antal annotations med fejl: {annotation_errors} (annotations i alt: {annotation_counter})')
print(f'Antal bilou fejl: {bilou_error_count}')
