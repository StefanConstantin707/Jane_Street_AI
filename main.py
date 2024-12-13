import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import pandas as pd
import polars as pl

from sklearn.metrics import mean_squared_error

from Models.MiniAttention import FullTransformer, FullDualTransformer
from Models.SimpleNN import SimpleNN
from TrainClass import TrainClass, EvalClass, MiniEvalClass
from Utillity.LossFunctions import r2_score, r2_loss
from dataHandler import create_dataloaders, load_data, split_data, load_data_symbol, create_seq_dataloaders, \
    load_data_xy_symbol, split_data_train_eval, create_day_data_loader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 20
    best = float('-inf')
    degraded = 0
    mini_epoch_size = 200
    dim_out = 9
    batch_size = 256
    lr = 3e-4

    model = FullTransformer(dim_in=88, dim_attn=64, attention_depth=1, mlp_depth=2, dim_out=dim_out, rotary_emb=True,
                            dropout=0.3, heads=1)
    best_model = model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    model = model.to(device)

    XY, time, weights = load_data_xy_symbol(0)
    train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights = split_data_train_eval(XY, time, weights, 0.9, 0.1)

    train_loader, eval_loader = create_day_data_loader(train_XY, train_time, train_weights, eval_XY, eval_time, eval_weights, batch_size, True,2)

    print("Data Loaded!")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction='none')

    miniEval = MiniEvalClass(model, eval_loader, optimizer, loss_function, device, mini_epoch_size, dim_out, 1)
    trainClass = TrainClass(model, train_loader, optimizer, loss_function, device, mini_epoch_size, dim_out, batch_size, miniEval)
    evalClass = EvalClass(model, eval_loader, optimizer, loss_function, device, mini_epoch_size, dim_out, 1)

    for epoch in range(epochs):
        train_loss, train_r2 = trainClass.step_epoch()
        eval_loss, eval_r2 = evalClass.step_eval()

        print(f'epoch {epoch} train loss {train_loss:.4f}, train_r2 {train_r2:.4f}, eval_loss {eval_loss:.4f}, eval_r2 {eval_r2:.4f}')
        if eval_r2 > best:
            best = eval_r2
            best_model = copy.deepcopy(model)
            model.save()
            degraded = 0
        else:
            degraded += 1
        if degraded > 5:
            break

if __name__ == '__main__':
    main()
