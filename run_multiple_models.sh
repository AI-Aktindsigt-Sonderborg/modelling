#python train_mlm_unsupervised_dp.py --max_length 16 &&
#python train_mlm_unsupervised_dp.py --max_length 32 &&
#python train_mlm_unsupervised_dp.py --max_length 64 &&

#python train_mlm_unsupervised_dp.py  --epsilon 2 &&
#python train_mlm_unsupervised_dp.py  --epsilon 8 &&
#python train_mlm_unsupervised_dp.py  --epsilon 16 &&
#python train_mlm_unsupervised_dp.py  --epsilon 200 &&

#python train_mlm_unsupervised.py --max_length 64 --train_data train_1000.json --epsilon 4 --lot_size 36 --evaluate_steps 100 && # 400 epochs
python train_mlm_unsupervised.py --max_length 64 --train_data train_5000.json --epsilon 4 --lot_size 36 --evaluate_steps 500 --epochs 80 &&
python train_mlm_unsupervised.py --max_length 64 --train_data train_10000.json --epsilon 4 --lot_size 36 --evaluate_steps 1000 --epochs 40

#python train_mlm_unsupervised.py --max_length 16 --epsilon 4 --lot_size 36 &&
#python train_mlm_unsupervised.py --max_length 32 --epsilon 4 --lot_size 36 &&
#python train_mlm_unsupervised.py --max_length 128 --epsilon 4 --lot_size 36 &&
#python train_mlm_unsupervised.py --max_length 256 --epsilon 4 --lot_size 36