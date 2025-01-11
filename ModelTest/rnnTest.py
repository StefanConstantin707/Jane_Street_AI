import torch
from torch import optim

from Models.SimpleNN import SimpleRNN
from TrainClass import GPUTrainEvalClass
from Utillity.LossFunctions import r2_loss
from dataHandlers.PartialDataHandler import SingleRowPD
from dataHandlers.SequencialDataHandler import GPULoaderLastTwenty


def rnn_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 79
    hidden_size = 256
    out_size = 9
    num_layers = 2

    dropout = 0.3
    noise = 1

    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4

    model = SimpleRNN(input_size=in_size, hidden_dim=hidden_size, output_size=out_size, num_layers=num_layers, dropout_prob=dropout, noise=noise, batch_norm=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    trainDataset = SingleRowPD(data_type='train', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False, dual_loading=False, sort_symbols=True)
    evalDataset = SingleRowPD(data_type='eval', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False, dual_loading=False, sort_symbols=True)

    train_loader = GPULoaderLastTwenty(trainDataset, True, batch_size, device, min_row_offset=1)
    eval_loader = GPULoaderLastTwenty(evalDataset, False, batch_size, device, min_row_offset=1)

    trainClass = GPUTrainEvalClass(model, train_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size, "train")
    evalClass = GPUTrainEvalClass(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size, "eval")

    for epoch in range(epochs):
        train_loss, train_r2_score, train_mse, train_r2_scores_responders = trainClass.step_epoch()

        # Print the general statistics
        print(
            f'Train statistics: Epoch: {epoch} Loss: {train_loss:.4f}, R2 score: {train_r2_score:.4f}, MSE: {train_mse:.4f}')

        # Print the R² scores for each responder
        for i, r2_score_responder in enumerate(train_r2_scores_responders):
            print(f'R2 score responder {i}: {r2_score_responder:.4f}')

        eval_loss, eval_r2_score, eval_mse, eval_r2_score_responders = evalClass.step_epoch()

        # Print the general statistics
        print(
            f'Eval statistics: Epoch: {epoch} Loss: {eval_loss:.4f}, R2 score: {eval_r2_score:.4f}, MSE: {eval_mse:.4f}')

        # Print the R² scores for each responder
        for i, eval_r2_score_responder in enumerate(eval_r2_score_responders):
            print(f'R2 score responder {i}: {eval_r2_score_responder:.4f}')