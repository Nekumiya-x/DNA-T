# DNA
# MIMIC
python -u run.py --data MIMIC --is_training 1  --model DNA --max_len 140 --e_layers 2 --d_model 64 --d_ff 128  --enc_in 20  --itr 5  --train_epochs 40 --gpu 0
# HA
python -u run.py --data HA --is_training 1  --model DNA --max_len 176 --e_layers 2 --d_model 64 --d_ff 128  --enc_in 7  --itr 5  --train_epochs 40 --gpu 0
# PWH
python -u run.py --data PWH --is_training 1  --model DNA --max_len 217 --e_layers 2 --d_model 128 --d_ff 256  --enc_in 7  --itr 5  --train_epochs 40 --gpu 0



