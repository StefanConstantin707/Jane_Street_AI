import numba
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

        self.train_eval_type = train_eval_type

        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

        self.total_iterations = len(self.loader)

        self.batch_size = batch_size
        self.mini_epoch_size = mini_epoch_size
        self.out_size = out_size

        self.epoch_loss = 0
        self.mini_epoch_loss = 0

        self.all_pred = []
        self.all_targets = []
        self.all_weights = []

        self.iteration = 0
        self.len_eval = self.loader.dataset.nu_rows

        self.stats_day_mapping = np.empty((0,))
        if train_eval_type == 'eval':
            self._create_day_mapping()

    def _reset_cache(self):
        self.all_pred = torch.zeros((self.len_eval, self.out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.len_eval, self.out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.len_eval, 1), dtype=torch.float32)
        self.all_temporal = torch.zeros((self.len_eval, 2), dtype=torch.float32)
        self.epoch_loss = 0
        self.mini_epoch_loss = 0
        self.iteration = 0
        self.last_stats_row = 0

    def _update_cache(self, y_batch, weights_batch, loss, outputs, temporal_batch):
        self.epoch_loss += loss.item()
        self.mini_epoch_loss += loss.item()

        start_idx = self.last_stats_row
        end_idx = self.last_stats_row + y_batch.shape[0]

        self.all_pred[start_idx:end_idx] = outputs
        self.all_targets[start_idx:end_idx] = y_batch
        self.all_weights[start_idx:end_idx] = weights_batch
        self.all_temporal[start_idx:end_idx] = temporal_batch

        self.last_stats_row = end_idx


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
            r2_score(self.all_pred[:, i], self.all_targets[:, i], self.all_weights.squeeze())
            for i in range(9)
        ]

        if self.train_eval_type == "eval":
            self._calculate_day_statistics()

        return avg_epoch_loss, avg_epoch_r2, avg_epoch_mse, avg_epoch_r2_responders

    def _create_day_mapping(self):
        @numba.njit()
        def _create_day_mapping_numba(dates, day_map):
            c_day = dates[0]
            last_index = 0
            c_day_idx = 0
            idx = 0
            for idx in range(dates.shape[0]):
                if c_day != dates[idx]:
                    c_day = dates[idx]
                    day_map[c_day_idx] = np.array([last_index, idx])
                    c_day_idx += 1
                    last_index = idx
            day_map[c_day_idx] = np.array([last_index, idx+1])
            return day_map

        all_temporal = self.loader.dataset.get_temporal().to(torch.int32).cpu().numpy()
        nu_days = self.loader.dataset.nu_days
        self.stats_day_mapping = np.zeros((nu_days, 2), dtype=np.int32)

        self.stats_day_mapping = _create_day_mapping_numba(all_temporal[:, 0], self.stats_day_mapping)

    def _calculate_day_statistics(self):
        r2_score_responder_6_per_day = []

        pred = self.all_pred[:, 6]
        target = self.all_targets[:, 6]
        weights = self.all_weights.squeeze()

        for i in range(self.stats_day_mapping.shape[0]):
            start_idx = self.stats_day_mapping[i, 0]
            end_idx = self.stats_day_mapping[i, 1]
            day_loss = r2_score(pred[start_idx:end_idx], target[start_idx:end_idx], weights[start_idx:end_idx]).item()
            r2_score_responder_6_per_day.append(day_loss)

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

    def _calculate_time_id_statistics(self):
        r2_score_per_day = []
        for i in range(978):
            mask = self.all_temporal[:, 1] == i
            pred = self.all_pred[:, 6][mask]
            target = self.all_targets[:, 6][mask]
            weights = self.all_weights[mask]

            r2_score_per_day.append(r2_score(pred, target, weights).item())
        plt.plot(r2_score_per_day)
        plt.show()

    def step_epoch(self):
        if self.train_eval_type == "eval":
            self.model.eval()
        elif self.train_eval_type == "train":
            self.model.train()

        self._reset_cache()

        for X_batch, Y_batch, temporal_batch, weights_batch, symbol_batch in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            temporal_batch = temporal_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            loss, outputs = self._run_model(X_batch, Y_batch, weights_batch)
            self._update_cache(Y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

        return self._calculate_statistics()

    def _run_model(self, x_batch, y_batch, weights_batch):
        if self.train_eval_type == "eval":
            with torch.no_grad():
                # Forward pass
                outputs = self.model(x_batch)

                if outputs.shape != y_batch.shape:
                    raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, y_batch.shape))

                loss = weighted_mse(y_true=y_batch, y_pred=outputs, weights=weights_batch)

                return loss, outputs
        elif self.train_eval_type == "train":
            outputs = self.model(x_batch)

            if outputs.shape != y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, y_batch.shape))

            loss = weighted_mse(y_true=y_batch, y_pred=outputs, weights=weights_batch)

            # Backpropagation
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # Reset gradient to 0
            self.optimizer.zero_grad()

            return loss, outputs


class GPUTrainEvalClass(GeneralTrainEvalClass):
    def __init__(self, model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, train_eval_type):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, train_eval_type)

    def step_epoch(self):
        if self.train_eval_type == "eval":
            self.model.eval()
        elif self.train_eval_type == "train":
            self.model.train()

        self._reset_cache()

        for i in range(self.loader.nu_batches):
            self.iteration += 1
            x_batch, y_batch, temporal_batch, weights_batch, symbol_batch = self.loader.get_batch()

            loss, outputs = self._run_model(x_batch, y_batch, weights_batch)
            self._update_cache(y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

        return self._calculate_statistics()


class GPUEvalOnlineClass(GeneralTrainEvalClass):
    def __init__(self, model, loader, cache_loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size):
        super().__init__(model, loader, optimizer, loss_function, device, out_size, batch_size, mini_epoch_size, "eval")

        self.cache_loader = cache_loader
        self.len_eval = self.loader.dataset.nu_rows
        self.last_stats_row = 0

    def step_epoch(self):

        self._reset_cache()

        for i in range(self.loader.nu_batches):
            self.model.eval()
            self.iteration += 1
            batch, x_batch, y_batch, temporal_batch, weights_batch, symbol_batch = self.loader.get_batch()
            self.cache_loader.add_data(batch)

            loss, outputs = self._run_model(x_batch, y_batch, weights_batch)
            self._update_cache(y_batch, weights_batch, loss, outputs, temporal_batch)
            self._log()

            self._online_train_step()

        return self._calculate_statistics()

    def _online_train_step(self):
        self.model.train()
        batch, x_batch, y_batch, temporal_batch, weights_batch, symbol_batch = self.cache_loader.get_batch()

        outputs = self.model(x_batch)

        if outputs.shape != y_batch.shape:
            raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, y_batch.shape))

        loss = weighted_mse(y_true=y_batch, y_pred=outputs, weights=weights_batch)

        # Backpropagation
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        # Reset gradient to 0
        self.optimizer.zero_grad()
