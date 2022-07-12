# semantic-modelling
- Clone repository to relevant directory on PW GPU
- Create python==3.8 env
- run `pip install -r requirements.txt` from <project_dir>
- run setup.py to create relevant directories etc.
- fetch data from /srv/aktio/deploy/web-data-scraper/python_api'
  - fx from terminal: `cp /srv/aktio/deploy/web-data-scraper
/python_api/*scrape_output* <path_to_project_dir>/data/new_scrape`
- Go to data_utils and run: `python preprocess_public_scraped_data.py 
--train_outfile <train_file_name> --val_outfile <val_file_name>`
- run train_mlm_unsupervised.py <args> to train mlm without differential 
privacy (see utils.input_args for available parameters): 
  - example: `python train_mlm_unsupervised.py --train_data <train_file_name> 
--eval_data <val_file_nam>`
