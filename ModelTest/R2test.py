import torch

from TrainClass import GeneralEval
from Utillity.LossFunctions import r2_score, weighted_mse
from dataHandlers.PartialDataHandler import BridgeDayDataset, SingleRowDataset


def r2_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_size = 167
    out_size = 9
    batch_size = 4096
    noise = 0.7

    evalDataset = SingleRowDataset(data_type='eval', path="./", start_date=1650, end_date=1699, out_size=out_size,
                                   in_size=in_size, device=device)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True, drop_last=True)
    total_iterations = len(eval_loader)
    all_pred = torch.zeros((total_iterations, batch_size, out_size), dtype=torch.float32)
    all_targets = torch.zeros((total_iterations, batch_size, out_size), dtype=torch.float32)
    all_weights = torch.zeros((total_iterations, batch_size, 1), dtype=torch.float32)

    iteration = 0
    for X_batch, Y_batch, temporal_batch, weights_batch, symbol_batch in eval_loader:
        Y_noise = Y_batch + torch.randn_like(Y_batch) * noise
        r2_loss = r2_score(Y_noise, Y_batch, weights_batch)

        all_pred[iteration - 1] = Y_noise
        all_targets[iteration - 1] = Y_batch
        all_weights[iteration - 1] = weights_batch

        iteration += 1
    r2_mean = r2_score(all_pred, all_targets, all_weights)
    mse_mean = weighted_mse(all_pred, all_targets, all_weights)
    print(r2_mean, mse_mean)