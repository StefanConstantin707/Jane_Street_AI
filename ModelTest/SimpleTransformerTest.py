import torch
from torch import optim, nn

from Models.MiniAttention import FullTransformer, FullSimpleTransformer
from TrainClass import GeneralTrain, GeneralEval, SequenceTrain, SequenceEval
from Utillity.LossFunctions import weighted_mse
from dataHandlers.SingleSymbolHandler import SequenceDataset, SequenceFeatureDataset


def attention_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_size = 79
    out_size = 9
    epochs = 20
    mini_epoch_size = 100
    batch_size = 1024
    lr = 3e-4
    noise = 5

    model = FullSimpleTransformer(dim_in=in_size, dim_attn=128, attention_depth=1, dim_out=9, mlp_dim=128, mlp_depth=2, rotary_emb=True, dropout=0.3, noise=noise).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss(reduction='none')

    trainDataset = SequenceFeatureDataset(data_type='train', path="./", start_date=0, end_date=1580, out_size=out_size, in_size=in_size, device=device, seq_len=2, symbol_id=0)
    evalDataset = SequenceFeatureDataset(data_type='eval', path="./", start_date=1580, end_date=1699, out_size=out_size, in_size=in_size, device=device, seq_len=2, symbol_id=0)

    train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(evalDataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    trainClass = SequenceTrain(model, train_loader, optimizer, weighted_mse, device, out_size, batch_size, mini_epoch_size)
    evalClass = SequenceEval(model, eval_loader, optimizer, weighted_mse, device, out_size, batch_size, mini_epoch_size)

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

