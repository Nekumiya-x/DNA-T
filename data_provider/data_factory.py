import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def get_data(data):
    pid, time, observation, mask, label, real_len, delta = [], [], [], [], [], [], []
    for patient in data:
        pid.append(int(patient[0]))
        time.append(patient[1])
        observation.append(patient[2])
        mask.append(patient[3])
        label.append(patient[4].cpu())
        delta.append(patient[5])
        real_len.append(patient[6])
    pid = torch.tensor(pid)
    time = torch.stack(time)
    observation = torch.stack(observation)
    mask = torch.stack(mask)
    label = torch.stack(label)
    real_len = torch.tensor(real_len)
    delta = torch.stack(delta)

    return pid, time, observation, mask, label, real_len, delta



def get_dataloader(args, train, valid, test):
    train_pid, train_time, train_observation, train_mask, train_label, train_real_len, train_delta = get_data(train)
    valid_pid, valid_time, valid_observation, valid_mask, valid_label, valid_real_len, valid_delta = get_data(valid)
    test_pid, test_time, test_observation, test_mask, test_label, test_real_len, test_delta = get_data(test)

    train_dataset = torch.utils.data.TensorDataset(train_observation, train_time, train_mask, train_delta, train_label,train_real_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)

    valid_dataset = torch.utils.data.TensorDataset(valid_observation, valid_time, valid_mask, valid_delta, valid_label,valid_real_len)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True)

    test_dataset = torch.utils.data.TensorDataset(test_observation, test_time, test_mask, test_delta, test_label,test_real_len)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader


def data_provider(args):
    # load pre-processed data
    total_dataset = torch.load(os.path.join(args.root_path, args.data, 'data.pt'))

    train_valid, test = train_test_split(total_dataset, test_size=0.1, random_state=0, shuffle=True)
    train, valid = train_test_split(train_valid, test_size=1 / 9, random_state=0, shuffle=True)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args, train, valid, test)

    return train_dataloader, valid_dataloader, test_dataloader

