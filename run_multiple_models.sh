python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --max_length 16 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --max_length 32 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --max_length 64 &&

python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 2 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 8 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 16 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 200 &&

python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 1 --lot_size 16 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 1 --lot_size 32 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 1 --lot_size 64 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 1 --lot_size 256 &&
python train_mlm_unsupervised_dp.py --train_data train_new_scrape.json --eval_data val_new_scrape.json --epsilon 1 --lot_size 512
