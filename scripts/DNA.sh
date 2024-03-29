# DNA
# MIMIC
python -u run.py --data MIMIC --is_training 1  --model DNA --max_len 140  --d_model 64 --d_ff 128  --enc_in 20  --train_epochs 40 --gpu 0
# HA
python -u run.py --data HA --is_training 1  --model DNA --max_len 176 --d_model 64 --d_ff 128  --enc_in 7   --train_epochs 40 --gpu 0
# PWH
python -u run.py --data PWH --is_training 1  --model DNA --max_len 217  --d_model 128 --d_ff 256  --enc_in 7   --train_epochs 40 --gpu 0



