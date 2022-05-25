# semantic-modelling
- Create python==3.8 env
- pip install -r requirements.txt
- run setup.py to create relevant directories
- run train_mlm_unsupervised.py to train mlm without differential privacy:
  - default parameters: python train_mlm_unsupervised.py --batch_size 16 --data_file da_DK_subset.json --delta 2e-05 --epochs 20 --epsilon 1, evaluate_steps=200, logging_steps=100, lot_size=16, lr=2e-05, max_grad_norm=1.2 max_length 100 --mlm_prob 0.15 --model_name jonfd/electra-small-nordic --save_steps 1000 --use_fp16 False --weight_decay 0.01 --whole_word_mask True
  - batch_size 128 is suitable for PW GPU
  - 