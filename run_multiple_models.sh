#python train_mlm_unsupervised_dp.py --max_length 16 &&
#python train_mlm_unsupervised_dp.py --max_length 32 &&
#python train_mlm_unsupervised_dp.py --max_length 64 &&

#python train_mlm_unsupervised_dp.py  --epsilon 2 &&
#python train_mlm_unsupervised_dp.py  --epsilon 8 &&
#python train_mlm_unsupervised_dp.py  --epsilon 16 &&
#python train_mlm_unsupervised_dp.py  --epsilon 200 &&


python train_mlm_unsupervised.py --max_length 32 --epsilon 2 --lot_size 18 --train_batch_size 6 &&
python train_mlm_unsupervised.py --max_length 32 --epsilon 2 --lot_size 36 --train_batch_size 6 &&
python train_mlm_unsupervised.py --max_length 32 --epsilon 2 --lot_size 72 --train_batch_size 6 &&
python train_mlm_unsupervised.py --max_length 32 --epsilon 2 --lot_size 144 --train_batch_size 6 &&
python train_mlm_unsupervised.py --max_length 32 --epsilon 2 --lot_size 288 --train_batch_size 6
