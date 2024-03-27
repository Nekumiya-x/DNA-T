import argparse
import datetime
import os
import torch
import random
import numpy as np

from exp.exp_main import Exp_Main




def main():

    manualSeed = 2023
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    parser = argparse.ArgumentParser(description='DNA')
    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model', type=str, required=True, default='DNA', help='model name')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='MIMIC', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--max_len', type=int, default=140, help='input sequence length')
    # model define
    parser.add_argument('--enc_in', type=int, default=20, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of Feed Forward Neural Network')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--DNA_size', type=int, default=5, help='DAN-Size,options:3,5,7,9,11')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=40, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--lr_ratio', type=float, default=0.2)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')


    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    for ii in range(args.itr):

        if args.is_training:
            # setting record of experiments
            setting = 'model_{}_data_{}_{}'.format(args.model, args.data,ii)

            exp = Exp(args)  # set experiments

            exp.train(setting)


            exp.test(setting,test=True)

            torch.cuda.empty_cache()
        else:
            ii = 0
            setting = 'model_{}_data_{}_{}'.format(args.model, args.data,ii)
            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting,test=True)

            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
