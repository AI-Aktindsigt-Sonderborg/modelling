# semantic-modelling
## Setup
- Clone repository to relevant directory on aktio GPU
- Create python==3.8 env (`conda create -n <repo_name> python==3.8`)
- Install cuda <u>if</u> not already installed:
  - via https://pytorch.org/get-started/locally/ and conda.
  - Find cuda version by running  `nvidia-smi` in terminal and then:
  - `conda install pytorch torchvision torchaudio cudatoolkit=<cuda_version> -c pytorch`
- run `pip install -r requirements.txt` from <project_dir>
- run setup.py to create relevant directories etc.
## This part is only relevant if data is not yet preprocessed
If in need for raw data contact Aktio. 
### Data generation for Tværkommunal Sprogmodel (Del 1):
The module `mlm.data_utils.RawScrapePreprocessing` is used to generate training and validation data. 
To see possible CLI options for this, run 

  `python -m mlm.data_utils.prep_scrape --help`
Example call: 

  `python -m mlm.data_utils.prep_scrape --train_outfile <train_file_name> --val_outfile <val_file_name> --split <train_size>`
### Data generation for Tværkommunal Sprogmodel (Del 2):
The module `sc.data_utils.ClassifiedScrapePreprocessing` is used to generate training and validation data. 
To see possible CLI options for this, run

  `python -m sc.data_utils.prep_scrape --help`
Example call:

  `python -m sc.data_utils.prep_scrape --train_outfile <train_file_name> --val_outfile <val_file_name> --split <train_size>`
### Data generation for Named Entity Recognition (AAPDA):
The module `ner.data_utils.prep_data` is used to generate training and validation data. 
To see possible CLI options for this, run
  `python -m ner.data_utils.prep_data --help`
Example call:
  `python -m ner.data_utils.prep_data --origin_input_file <name_of_input_file> --train_outfile <train_file_name> --val_outfile <validation_file_name> --split <train_size>`

## MLM training
To see possible CLI options for this, run
  `python -m mlm.train --help`
Example call:
  `python -m mlm.train --train_data train.jsonl --eval_data validation.jsonl`
## Sequence Classification training
To see possible CLI options for this, run
  `python -m sc.train --help`
Example call:
  `python -m sc.train --train_data train.jsonl --eval_data validation.jsonl`