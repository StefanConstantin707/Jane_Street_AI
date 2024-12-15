import time

import torch
from torch import optim, nn

from Models.SimpleNN import SimpleNN
from TrainClass import GeneralTrain, GeneralEval
from Utillity.LossFunctions import weighted_mse
from dataHandlers.PartialDataHandler import SingleRowDataset


def nn_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    in_size = 80
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4

    model = SimpleNN(input_size=in_size, hidden_dim=64, output_size=out_size, num_layers=5, dropout_prob=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = SingleRowDataset(data_type='train', path="./", start_date=1400, end_date=1580, data_percentage=0.9, out_size=out_size, in_size=in_size, device=device)
    evalDataset = SingleRowDataset(data_type='eval', path="./", start_date=1580, end_date=1699, data_percentage=0.9, out_size=out_size, in_size=in_size, device=device)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    trainClass = GeneralTrain(model, train_loader, optimizer, weighted_mse, device, out_size, batch_size, mini_epoch_size)
    evalClass = GeneralEval(model, eval_loader, optimizer, weighted_mse, device, out_size, batch_size, mini_epoch_size)

    for epoch in range(epochs):
        train_loss, train_r2_score, train_mse, train_r2_score_responder_six = trainClass.step_epoch()
        print(f'Train statistics: Epoch: {epoch} Loss: {train_loss:.4f}, R2 score: {train_r2_score:.4f}, MSE: {train_mse:.4f}, R2 score responder6: {train_r2_score_responder_six:.4f}')

        eval_loss, eval_r2_score, eval_mse, eval_r2_score_responder_six = evalClass.step_epoch()
        print(
            f'Eval statistics: Epoch: {epoch} Loss: {eval_loss:.4f}, R2 score: {eval_r2_score:.4f}, MSE: {eval_mse:.4f}, R2 score responder6: {eval_r2_score_responder_six:.4f}')

