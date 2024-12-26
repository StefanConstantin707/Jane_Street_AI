import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
from Utillity.LossFunctions import r2_loss, r2_score, weighted_mse, weighted_mse_r6, weighted_mse_r6_weighted
from scipy.stats import linregress


class GeneralTrainEvalClass:
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, train_eval_type):
        super().__init__()

        self.model = model
        self.model.train()

        self.train_eval_type = train_eval_type

        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

        self.total_iterations = 0

        self.batch_size = batch_size
        self.mini_epoch_size = mini_epoch_size
        self.out_size = out_size

        self.epoch_loss = 0
        self.mini_epoch_loss = 0

        self.all_pred = []
        self.all_targets = []
        self.all_weights = []

        self.iteration = 0

    def _reset_cache(self):
        self.all_pred = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.total_iterations, self.batch_size, 1), dtype=torch.float32)
        self.all_temporal = torch.zeros((self.total_iterations, self.batch_size, 2), dtype=torch.float32)
        self.epoch_loss = 0
        self.mini_epoch_loss = 0
        self.iteration = 0

    def _update_cache(self, Y_batch, weights_batch, loss, outputs, temporal_batch):
        self.epoch_loss += loss.item()
        self.mini_epoch_loss += loss.item()

        self.all_pred[self.iteration - 1, :outputs.shape[0]] = outputs
        self.all_targets[self.iteration - 1, :outputs.shape[0]] = Y_batch
        self.all_weights[self.iteration - 1, :outputs.shape[0]] = weights_batch
        self.all_temporal[self.iteration - 1, :outputs.shape[0]] = temporal_batch

    def _log(self):
        if self.iteration % self.mini_epoch_size == 0 or self.iteration == self.total_iterations:
            nu_iterations_since_last_log = min(self.mini_epoch_size, ((self.iteration - 1) % self.mini_epoch_size) + 1)
            print(
                f"Iteration {self.iteration}/{self.total_iterations}, Average_Loss: {self.mini_epoch_loss / nu_iterations_since_last_log:.4f}")
            self.mini_epoch_loss = 0

    def _calculate_statistics(self):
        avg_epoch_loss = self.loss_function(self.all_pred, self.all_targets, self.all_weights)
        avg_epoch_r2 = r2_score(self.all_pred, self.all_targets, self.all_weights)
        avg_epoch_mse = weighted_mse(self.all_pred, self.all_targets, self.all_weights)

        avg_epoch_r2_responders = [
            r2_score(self.all_pred[:, :, i], self.all_targets[:, :, i], self.all_weights.squeeze())
            for i in range(9)
        ]

        # r2_score_per_day = []
        # for i in range(978):
        #     mask = self.all_temporal[:, :, 1].view(-1) == i
        #     pred = self.all_pred[:, :, 6].view(-1)[mask]
        #     target = self.all_targets[:, :, 6].view(-1)[mask]
        #     weights = self.all_weights.view(-1)[mask]
        #
        #     r2_score_per_day.append(r2_score(pred, target, weights=weights).item())
        # plt.plot(r2_score_per_day)
        # plt.show()

        if self.train_eval_type == "eval":
            self._calculate_day_statistics()

        return avg_epoch_loss, avg_epoch_r2, avg_epoch_mse, avg_epoch_r2_responders

    def _calculate_day_statistics(self):
        r2_score_responder_6_per_day = []
        all_temporal = self.all_temporal.view(-1, 2)

        pred = self.all_pred[:, :, 6].view(-1)
        target = self.all_targets[:, :, 6].view(-1)
        weights = self.all_weights.view(-1)

        c_day = all_temporal[0, 0]
        prev_idx = 0
        for i in range(all_temporal.shape[0]):
            if all_temporal[i, 0] != c_day:
                c_day = all_temporal[i, 0]
                day_loss = r2_score(pred[prev_idx:i], target[prev_idx:i], weights[prev_idx:i]).item()
                r2_score_responder_6_per_day.append(day_loss)
                prev_idx = i

        # Convert to numpy array for regression analysis
        days = np.arange(len(r2_score_responder_6_per_day))
        r2_scores = np.array(r2_score_responder_6_per_day)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(days, r2_scores)

        # Print regression statistics
        print("Linear Regression Statistics:")
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value ** 2}")
        print(f"P-value: {p_value}")
        print(f"Standard Error: {std_err}")

        # Plot the data and regression line
        plt.figure(figsize=(10, 6))
        plt.plot(days, r2_scores, label='R2 Scores', marker='o')
        plt.plot(days, slope * days + intercept, label='Regression Line', color='red', linestyle='--')
        plt.xlabel('Day Index')
        plt.ylabel('R2 Score')
        plt.title('R2 Score Responder 6 Per Day with Regression Line')
        plt.legend()
        plt.show()


class GeneralTrain(GeneralTrainEvalClass):
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, "train")
        self.total_iterations = len(self.loader)

    def step_epoch(self):
        self.model.train()
        self._reset_cache()
        for X_batch, Y_batch, temporal_batch, weights_batch, symbol_batch in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)
            symbol_batch = symbol_batch.to(self.device)

            loss, outputs = self._run_model_with_optimization(X_batch, Y_batch, weights_batch)
            self._update_cache(Y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

        return self._calculate_statistics()

    def _run_model_with_optimization(self, X_batch, Y_batch, weights_batch):
        # Forward pass
        outputs = self.model(X_batch)

        if outputs.shape != Y_batch.shape:
            raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

        loss = weighted_mse(y_true=Y_batch, y_pred=outputs, weights=weights_batch)

        # Backpropagation
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        # Reset gradient to 0
        self.optimizer.zero_grad()

        return loss, outputs


class GeneralEval(GeneralTrainEvalClass):
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, "eval")
        self.total_iterations = len(self.loader)

    def step_epoch(self):
        self.model.eval()
        self._reset_cache()
        for X_batch, Y_batch, temporal_batch, weights_batch, symbol_batch in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)
            symbol_batch = symbol_batch.to(self.device)

            loss, outputs = self._run_model(X_batch, Y_batch, weights_batch)
            self._update_cache(Y_batch, weights_batch, loss, outputs, temporal_batch)

        return self._calculate_statistics()

    def _run_model(self, X_batch, Y_batch, weights_batch):
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X_batch)

            if outputs.shape != Y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

            loss = weighted_mse(y_true=Y_batch, y_pred=outputs, weights=weights_batch)

            return loss, outputs

class SequenceTrain(GeneralTrainEvalClass):
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size)
        self.total_iterations = len(self.loader)

    def step_epoch(self):
        self.model.train()
        self._reset_cache()
        for X_batch, Y_batch, temporal_batch, weights_batch in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            loss, outputs = self._run_model_with_optimization(X_batch, Y_batch, weights_batch)
            self._update_cache(Y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

        return self._calculate_statistics()

    def _run_model_with_optimization(self, X_batch, Y_batch, weights_batch):
        # Forward pass
        outputs = self.model(X_batch)

        if outputs.shape != Y_batch.shape:
            raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

        loss = weighted_mse(y_true=Y_batch, y_pred=outputs, weights=weights_batch)

        # Backpropagation
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        # Reset gradient to 0
        self.optimizer.zero_grad()

        return loss, outputs

class SequenceEval(GeneralTrainEvalClass):
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size)
        self.total_iterations = len(self.loader)

    def step_epoch(self):
        self.model.eval()
        self._reset_cache()
        for X_batch, Y_batch, temporal_batch, weights_batch in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            loss, outputs = self._run_model(X_batch, Y_batch, weights_batch)
            self._update_cache(Y_batch, weights_batch, loss, outputs, temporal_batch)

        return self._calculate_statistics()

    def _run_model(self, X_batch, Y_batch, weights_batch):
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X_batch)

            if outputs.shape != Y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

            loss = weighted_mse(y_true=Y_batch, y_pred=outputs, weights=weights_batch)

            return loss, outputs
