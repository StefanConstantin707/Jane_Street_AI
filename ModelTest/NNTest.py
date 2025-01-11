import torch
from torch import optim, nn

from Models.SimpleNN import SimpleNN
from TrainClass import GPUTrainEvalClass, GeneralTrainEvalClass, GPUEvalOnlineClass, GPUReplaceTrainData
from Utillity.LossFunctions import r2_loss
from dataHandlers.DataHandlerGeneral import GPULoaderGeneral
from dataHandlers.GPULoaders import GPULoaderResponderLag, GPUOnlineCacheLoader, GPUOnlineLoader, \
    GPUOnlineCacheLoaderSequential
from dataHandlers.PartialDataHandler import SingleRowPD
from dataHandlers.ResponderOnlyDataHandler import SingleRowPDResponder, GPULoaderResponder


def nn_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 79
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 1e-5
    noise = 1

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = SingleRowPD(data_type='train', path="./", start_date=1100, end_date=1580, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False)
    evalDataset = SingleRowPD(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

    trainClass = GeneralTrainEvalClass(model, train_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size, "train")
    evalClass = GeneralTrainEvalClass(model, eval_loader, optimizer, r2_loss, device, out_size, batch_size, mini_epoch_size, "eval")

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

def nn_test_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 80
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 4096
    lr = 1e-5
    noise = 1

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise, batch_norm=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    trainDataset = SingleRowPD(data_type='train', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False, sort_symbols=True, dual_loading=False)
    evalDataset = SingleRowPD(data_type='eval', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, collect_data_at_loading=False, normalize_features=False, sort_symbols=True, dual_loading=False)

    train_loader = GPULoaderResponderLag(trainDataset, True, batch_size, device)
    eval_loader = GPULoaderResponderLag(evalDataset, False, batch_size, device)

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

def nn_test_online():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 79
    out_size = 9
    mini_epoch_size = 100
    batch_size = 4096
    lr = 1e-5
    noise = 1

    model = SimpleNN(input_size=in_size, hidden_dim=256, output_size=out_size, num_layers=2, dropout_prob=0.3, noise=noise, batch_norm=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    model.load_state_dict(torch.load("./savedModels/nn_model_weights_0.008_0.0108max.pth", weights_only=True))

    online_optimizer = optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=1e-4)

    cacheDataset = SingleRowPD(data_type='train', path="./JaneStreetRealTimeMarketDataForecasting",
                               start_date=1500, end_date=1580, out_size=out_size, in_size=in_size, device=device,
                               collect_data_at_loading=False, normalize_features=False, sort_symbols=False,
                               dual_loading=False)
    evalDataset = SingleRowPD(data_type='eval', path="./JaneStreetRealTimeMarketDataForecasting",
                              start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device,
                              collect_data_at_loading=False, normalize_features=False, sort_symbols=False,
                              dual_loading=False)

    loader = GPULoaderGeneral(cacheDataset, False, batch_size, device)

    cache_loader = GPUOnlineCacheLoaderSequential(cacheDataset, True, batch_size, device)
    eval_loader = GPUOnlineLoader(evalDataset, device)

    replaceData = GPUReplaceTrainData(model, loader, batch_size, mini_epoch_size)
    replaceData.step_epoch()

    cache_loader.dataset.data = loader.dataset.data

    evalOnlineClass = GPUEvalOnlineClass(model, eval_loader, cache_loader, online_optimizer, r2_loss, device, out_size,
                                         batch_size, mini_epoch_size)

    print("Start eval")
    eval_loss, eval_r2_score, eval_mse, eval_r2_score_responders = evalOnlineClass.step_epoch()

    # Print the general statistics
    print(
        f'Eval statistics: Loss: {eval_loss:.4f}, R2 score: {eval_r2_score:.4f}, MSE: {eval_mse:.4f}')

    # Print the R² scores for each responder
    for i, eval_r2_score_responder in enumerate(eval_r2_score_responders):
        print(f'R2 score responder {i}: {eval_r2_score_responder:.4f}')

def nn_test_lag_responder():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 20
    hidden_size = 256
    out_size = 9
    num_layers = 4

    dropout = 0
    noise = 0

    epochs = 20
    mini_epoch_size = 1000
    batch_size = 4096
    lr = 3e-5

    model = SimpleNN(input_size=in_size, hidden_dim=hidden_size, output_size=out_size, num_layers=num_layers, dropout_prob=dropout, noise=noise, batch_norm=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    trainDataset = SingleRowPDResponder(data_type='train', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1000, end_date=1580, out_size=out_size, in_size=in_size, device=device, sort_symbols=True)
    evalDataset = SingleRowPDResponder(data_type='eval', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, sort_symbols=True)

    train_loader = GPULoaderResponder(trainDataset, True, batch_size, device, min_row_offset=30, max_row_offset=0)
    eval_loader = GPULoaderResponder(evalDataset, False, batch_size, device, min_row_offset=30, max_row_offset=0)

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

def nn_test_lag_responder_mapping():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    in_size = 159
    hidden_size = 256
    out_size = 9
    num_layers = 4

    dropout = 0
    noise = 1

    epochs = 20
    mini_epoch_size = 1000
    batch_size = 4096
    lr = 1e-5

    model = SimpleNN(input_size=in_size, hidden_dim=hidden_size, output_size=out_size, num_layers=num_layers, dropout_prob=dropout, noise=noise, batch_norm=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    trainDataset = SingleRowPD(data_type='train', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1400, end_date=1580, out_size=out_size, in_size=in_size, device=device, sort_symbols=True, collect_data_at_loading=False, dual_loading=False, normalize_features=False)
    evalDataset = SingleRowPD(data_type='eval', path="./JaneStreetRealTimeMarketDataForecasting", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, sort_symbols=True, collect_data_at_loading=False, dual_loading=False, normalize_features=False)

    train_loader = GPULoaderResponder(trainDataset, True, batch_size, device, min_row_offset=20, max_row_offset=0)
    eval_loader = GPULoaderResponder(evalDataset, False, batch_size, device, min_row_offset=20, max_row_offset=0)

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
