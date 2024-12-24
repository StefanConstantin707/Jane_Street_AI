import time

import torch
from torch import optim, nn

from Models.SimpleNN import SimpleNN, SimpleNNEmbed, SimpleNNSampling
from TrainClass import GeneralTrain, GeneralEval
from Utillity.LossFunctions import weighted_mse, r2_loss
from dataHandlers.PartialDataHandler import RowSamplerDatasetPD, SingleRowPD
from dataHandlers.SingleSymbolHandler import SingleRowSamplingDatasetSS


def nn_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 79
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4
    noise = 0.5

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = SingleRowPD(data_type='train', path="./", start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False)
    evalDataset = SingleRowPD(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

    trainClass = GeneralTrain(model, train_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)
    evalClass = GeneralEval(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)

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

def nn_test_prev_responder():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_size = 167
    out_size = 9
    epochs = 5
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4
    noise = 0.05

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = PrevResponderAndFeaturesDataset(data_type='train', path="./", start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, device=device)
    evalDataset = PrevResponderAndFeaturesDataset(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    trainClass = GeneralTrain(model, train_loader, optimizer, weighted_mse, device, out_size, batch_size, mini_epoch_size)
    evalClass = GeneralEval(model, eval_loader, optimizer, weighted_mse, device, out_size, batch_size, mini_epoch_size)

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

        model.save(experiment_id=37)

def nn_test_bridge_gap():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_size = 167
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4
    noise = 0.35

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3,
                     noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = BridgeDayDataset(data_type='train', path="./", start_date=700, end_date=1580, out_size=out_size,
                                    in_size=in_size, device=device)
    evalDataset = BridgeDayDataset(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size,
                                   in_size=in_size, device=device)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    trainClass = GeneralTrain(model, train_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)
    evalClass = GeneralEval(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)

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

def nn_test_f_and_responders():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_size = 88
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4
    noise = 0.01

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3,
                     noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = SingleRowRespDataset(data_type='train', path="./", start_date=1400, end_date=1580, out_size=out_size,
                                    in_size=in_size, device=device)
    evalDataset = SingleRowRespDataset(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size,
                                   in_size=in_size, device=device)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    trainClass = GeneralTrain(model, train_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)
    evalClass = GeneralEval(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)

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

        model.save(12345)

def nn_test_past_sampling():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_size = 89
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 3e-4
    noise = 0.35

    model = SimpleNNSampling(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = RowSamplerDatasetPD(data_type='train', path="./", start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, collect_data_at_loading=False, normalize_features=False, device=device)
    evalDataset = RowSamplerDatasetPD(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, collect_data_at_loading=False, normalize_features=False, device=device)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    trainClass = GeneralTrain(model, train_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)
    evalClass = GeneralEval(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size)

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
