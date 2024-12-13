import torch

from Utillity.LossFunctions import r2_loss, r2_score


class TrainClass:
    def __init__(self, model, loader, optimizer, loss_function, device, mini_epoch_size, out_size, batch_size, miniEval):
        super().__init__()

        self.model = model
        self.model.train()

        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

        self.total_iterations = len(self.loader)

        self.batch_size = batch_size
        self.out_size = out_size

        self.epoch_loss = 0
        self.mini_epoch_loss = 0
        self.mini_epoch_size = mini_epoch_size

        self.all_pred = torch.zeros((self.total_iterations, batch_size, out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.total_iterations, batch_size, out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.total_iterations, batch_size, 1), dtype=torch.float32)

        self.iteration = 0

        self.miniEval = miniEval

    def step_epoch(self):
        for X_batch, Y_batch, weights_batch, time in self.loader:
            self.iteration += 1

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            # Reset gradient to 0
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_batch)

            if outputs.shape != Y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

            loss = r2_loss(y_true=Y_batch, y_pred=outputs, weights=weights_batch).mean()

            # Backpropagation
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # Accumulate total loss
            self.epoch_loss += loss.item()
            self.mini_epoch_loss += loss.item()

            self.all_pred[self.iteration - 1, :outputs.shape[0]] = outputs
            self.all_targets[self.iteration - 1, :outputs.shape[0]] = Y_batch
            self.all_weights[self.iteration - 1, :outputs.shape[0]] = weights_batch

            # Log information every mini_epoch_size iterations
            if self.iteration % self.mini_epoch_size == 0 or self.iteration == self.total_iterations:
                eval_loss, eval_r2 = self.miniEval.step_eval()
                print(f"Iteration {self.iteration}/{self.total_iterations}, Average_Loss: {self.mini_epoch_loss / min(self.mini_epoch_size, ((self.iteration - 1) % self.mini_epoch_size) + 1):.4f}, eval_loss {eval_loss:.4f}, eval_r2 {eval_r2:.4f}")
                self.mini_epoch_loss = 0



        # Compute overall metrics
        avg_epoch_r2 = r2_score(self.all_targets, self.all_pred, self.all_weights)

        avg_epoch_loss = self.epoch_loss / self.total_iterations

        self.all_pred = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.total_iterations, self.batch_size, 1), dtype=torch.float32)
        self.iteration = 0
        self.epoch_loss = 0

        return avg_epoch_loss, avg_epoch_r2


class EvalClass:
    def __init__(self, model, loader, optimizer, loss_function, device, mini_epoch_size, out_size, batch_size):
        super().__init__()

        self.model = model
        self.model.eval()

        self.loader = loader
        self.loss_function = loss_function
        self.device = device

        self.batch_size = batch_size
        self.out_size = out_size

        self.total_iterations = len(self.loader)

        self.epoch_loss = 0

        self.all_pred = torch.zeros((self.total_iterations, batch_size, out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.total_iterations, batch_size, out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.total_iterations, batch_size, 1), dtype=torch.float32)

        self.iteration = 0

        self.prev_pred = torch.zeros((batch_size, out_size), dtype=torch.float32)
        self.prev_targets = torch.zeros((batch_size, out_size), dtype=torch.float32)

    def step_eval(self):
        for X_batch, Y_batch, weights_batch, time in self.loader:
            self.iteration += 1

            if time[:, 1] == 0:
                self.prev_pred = self.prev_targets

            X_batch[:, -1, -9:] = self.prev_pred

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            # Forward pass
            outputs = self.model(X_batch)

            self.prev_pred = outputs
            self.prev_targets = Y_batch

            if outputs.shape != Y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

            loss = r2_loss(y_true=Y_batch, y_pred=outputs, weights=weights_batch).mean()

            # Accumulate total loss
            self.epoch_loss += loss.item()

            self.all_pred[self.iteration - 1, :outputs.shape[0]] = outputs
            self.all_targets[self.iteration - 1, :outputs.shape[0]] = Y_batch
            self.all_weights[self.iteration - 1, :outputs.shape[0]] = weights_batch

        # Compute overall metrics
        avg_epoch_r2 = r2_score(self.all_targets, self.all_pred, self.all_weights)

        avg_epoch_loss = self.epoch_loss / self.total_iterations

        self.all_pred = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.total_iterations, self.batch_size, 1), dtype=torch.float32)
        self.iteration = 0
        self.epoch_loss = 0

        return avg_epoch_loss, avg_epoch_r2


class MiniEvalClass:
    def __init__(self, model, loader, optimizer, loss_function, device, mini_epoch_size, out_size, batch_size):
        super().__init__()

        self.model = model
        self.model.eval()

        self.loader = loader
        self.loss_function = loss_function
        self.device = device

        self.batch_size = batch_size
        self.out_size = out_size

        self.total_iterations = len(self.loader)

        self.epoch_loss = 0

        self.all_pred = torch.zeros((968, batch_size, out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((968, batch_size, out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((968, batch_size, 1), dtype=torch.float32)

        self.iteration = 0

        self.prev_pred = []

    def step_eval(self):

        self.all_pred = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_targets = torch.zeros((self.total_iterations, self.batch_size, self.out_size), dtype=torch.float32)
        self.all_weights = torch.zeros((self.total_iterations, self.batch_size, 1), dtype=torch.float32)
        self.iteration = 0
        self.epoch_loss = 0

        for X_batch, Y_batch, weights_batch, time in self.loader:
            if time[:, 1] == 967 and self.iteration == 0:
                self.prev_pred = Y_batch
                continue
            elif time[:, 1] != 0 and self.iteration == 0:
                continue

            self.iteration += 1

            X_batch[:, -1, -9:] = self.prev_pred

            # Move data to specified device (CPU or GPU)
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            # Forward pass
            outputs = self.model(X_batch)

            self.prev_pred = outputs

            if outputs.shape != Y_batch.shape:
                raise ValueError("Output shape mismatch, with shapes {} vs {} ".format(outputs.shape, Y_batch.shape))

            loss = r2_loss(y_true=Y_batch, y_pred=outputs, weights=weights_batch).mean()

            # Accumulate total loss
            self.epoch_loss += loss.item()

            self.all_pred[self.iteration - 1, :outputs.shape[0]] = outputs
            self.all_targets[self.iteration - 1, :outputs.shape[0]] = Y_batch
            self.all_weights[self.iteration - 1, :outputs.shape[0]] = weights_batch

            if time[:, 1] == 967:
                break

        # Compute overall metrics
        avg_epoch_r2 = r2_score(self.all_targets[:, :, 6:7], self.all_pred[:, :, 6:7], self.all_weights)

        avg_epoch_loss = self.epoch_loss / 968

        return avg_epoch_loss, avg_epoch_r2


