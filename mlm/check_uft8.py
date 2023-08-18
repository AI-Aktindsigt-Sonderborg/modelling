from mlm.data_utils.data_prep_input_args import DataPrepArgParser
from mlm.data_utils.prep_data import RawDataPreprocessing
from mlm.local_constants import PREP_DATA_DIR, CONF_DATA_DIR
from shared.utils.helpers import read_json_lines

prep_parser = DataPrepArgParser()
prep_args = prep_parser.parser.parse_args()


data = read_json_lines(CONF_DATA_DIR, prep_args.raw_file)

print(data[0]['pdf_text'])
# for obs in data:
#   entity = obs["entities"][0]
#   obs["tags"] = [tag if tag[2:] == entity else "O" for tag in obs["tags"]]
