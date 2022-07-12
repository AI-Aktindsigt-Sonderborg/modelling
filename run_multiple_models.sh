#python train_mlm_unsupervised_dp.py --max_length 16 &&
#python train_mlm_unsupervised_dp.py --max_length 32 &&
#python train_mlm_unsupervised_dp.py --max_length 64 &&

#python train_mlm_unsupervised_dp.py  --epsilon 2 &&
#python train_mlm_unsupervised_dp.py  --epsilon 8 &&
#python train_mlm_unsupervised_dp.py  --epsilon 16 &&
#python train_mlm_unsupervised_dp.py  --epsilon 200 &&

python train_mlm_unsupervised.py --max_length 64 --epsilon 1 --lot_size 36 &&
python train_mlm_unsupervised.py --max_length 64 --epsilon 2 --lot_size 36 &&
python train_mlm_unsupervised.py --max_length 64 --epsilon 4 --lot_size 36 &&
python train_mlm_unsupervised.py --max_length 64 --epsilon 8 --lot_size 36 &&
python train_mlm_unsupervised.py --max_length 64 --epsilon 16 --lot_size 36
