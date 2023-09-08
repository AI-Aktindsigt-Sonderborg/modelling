# semantic-modelling
## Setup
- Clone repository to relevant directory on aktio GPU
- Create python==3.8 env
- Install cuda IF not already installed: 
  - via https://pytorch.org/get-started/locally/ and conda.
  - Find cuda version by running  `nvidia-smi` in terminal and then:
  - `conda install pytorch torchvision torchaudio cudatoolkit=<cuda_version> -c pytorch`
- run `pip install -r requirements.txt` from <project_dir>
- run setup.py to create relevant directories etc.
## This part is only relevant if data is not yet preprocessed
If in need for raw data contact Aktio. 
### Data generation for Tværkommunal sprogmodel (Del 1):
  - The module `mlm.data_utils.RawScrapePreprocessing` is used to generate training and validation data. 
  To see possible CLI options for this, run
  
  `python -m mlm.data_utils.prep_scrape --help`

  - Example call: `python -m mlm.data_utils.prep_scrape
  --train_outfile <train_file_name> --val_outfile <val_file_name> --split <train_size>`
### Data generation for Tværkommunal sprogmodel (Del 2):
  - The module `sc.data_utils.ClassifiedScrapePreprocessing` is used to generate training and validation data. 
  To see possible CLI options for this, run
  
  `python -m sc.data_utils.prep_scrape --help`

  - Example call: `python -m sc.data_utils.prep_scrape
  --train_outfile <train_file_name> --val_outfile <val_file_name> --split <train_size>`


## MLM training
See modelling_utils.input_args for args.
- run train_mlm <args> to train mlm without differential 
privacy (see utils.input_args for available parameters): 
  - example: `python -m train_mlm --train_data <train_file_name> 
--eval_data <val_file_nam>`
## Sequence Classification training
See modelling_utils.input_args for args.
- run train_sc <args> to train sc model 
privacy (see utils.input_args for available parameters): 
  - example: `python -m train_sc --train_data <train_file_name> 
--eval_data <val_file_name>`
