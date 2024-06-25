import logging
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop

from sad.base.base_dataset import BaseADDataset
from sad.base.base_net import BaseNet
from sad.base.base_trainer import BaseTrainer

import wandb

class DeepSADTrainer(BaseTrainer):
    def __init__(
        self,
        c,
        eta: float,
        optimizer_name: str = "adam",
        lr: float = 0.001,
        n_epochs: int = 150,
        lr_milestones: tuple = (),
        batch_size: int = 128,
        weight_decay: float = 1e-6,
        device: str = "cuda",
        n_jobs_dataloader: int = 0,
        l1_reg: float = 0.0,
        crop_size: int = 0,
    ):
        super().__init__(
            optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device, n_jobs_dataloader
        )

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        if crop_size != 0:
            self.transform = RandomCrop(size=crop_size)
        else:
            self.transform = None

        # Optimization parameters
        self.eps = 1e-6
        self.l1_reg = l1_reg

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet, tolerable_fnr: list = [0.0, 0.001, 0.01, 0.005]):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info("Initializing center c...")
            self.c = self.init_center_c(train_loader, net)
            logger.info("Center c initialized.")

        # Training
        logger.info("Starting training...")
        start_time = time.time()
        for epoch in range(self.n_epochs):
            net.train()
            # scheduler.step()
            if epoch in self.lr_milestones:
                logger.info("  LR scheduler: new learning rate is %g" % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data

                if self.transform:
                    inputs = self.transform(inputs)
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                # Experiment L1-Regularization
                if self.l1_reg > 0.0:
                    l1_weights = torch.cat(
                        [
                            param.flatten()
                            for param_name, param in net.named_parameters()
                            if param_name.endswith("weight")
                        ]
                    )
                    loss += self.l1_reg * torch.norm(l1_weights, 1)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            # compute thresholds on validation set
            if (epoch + 1) % 10 == 0:
                net.eval()
                val_distances = []
                for batch in DataLoader(dataset.val_set, batch_size=self.batch_size):
                    if self.transform:
                        batch[0] = self.transform(batch[0])
                    output = net(batch[0].cuda()).detach()
                    distance = torch.sum((output - torch.Tensor(self.c)) ** 2, dim=1)
                    val_distances.append(distance)
                val_distances = torch.cat(val_distances)

                thresholds = {}
                for fnr in tolerable_fnr:
                    thresholds[f"fnr={fnr}"] = torch.quantile(val_distances, q=1 - fnr).item()

                # log epoch training accuracies
                train_distances = []
                y_train = []
                for batch in train_loader:
                    if self.transform:
                        batch[0] = self.transform(batch[0])
                    output = net(batch[0].cuda()).detach()
                    distance = torch.sum((output - torch.Tensor(self.c)) ** 2, dim=1)
                    train_distances.append(distance)
                    y_train.append(batch[1])
                train_distances = torch.cat(train_distances)
                y_train = torch.cat(y_train)

                accuracies = {}
                for threshold_name, threshold_value in thresholds.items():
                    y_train_pred = (
                        torch.where(train_distances > threshold_value, -1, 1).detach().cpu()
                    )  # greater -> -1 = outlier
                    accuracies[threshold_name] = accuracy_score(y_true=y_train, y_pred=y_train_pred)

                # log epoch statistics
                epoch_train_time = time.time() - epoch_start_time
                logger.info(
                    f"| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s "
                    f"| Train Loss: {epoch_loss / n_batches:.6f} "
                    f"| Train Accuracy: {accuracies}|"
                )
                if wandb.run is not None:
                    wandb.log({"epoch": epoch + 1,
                               "train_loss": epoch_loss / n_batches,
                                 "train_accuracy": accuracies
                               })
            else:
                # log epoch statistics
                epoch_train_time = time.time() - epoch_start_time
                logger.info(
                    f"| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s "
                    f"| Train Loss: {epoch_loss / n_batches:.6f} |"
                )

        self.train_time = time.time() - start_time
        logger.info("Training Time: {:.3f}s".format(self.train_time))
        logger.info("Finished training.")

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info("Starting testing...")
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                # inputs, semi_targets = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(
                    zip(
                        idx.cpu().data.numpy().tolist(),
                        labels.cpu().data.numpy().tolist(),
                        scores.cpu().data.numpy().tolist(),
                    )
                )

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = -1 * np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info("Test Loss: {:.6f}".format(epoch_loss / n_batches))
        logger.info("Test AUC: {:.2f}%".format(100.0 * self.test_auc))
        logger.info("Test Time: {:.3f}s".format(self.test_time))
        logger.info("Finished testing.")
        if wandb.run is not None:
            wandb.log({"test_loss": epoch_loss / n_batches,
                       "test_auc": 100.0 * self.test_auc})

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                if self.transform:
                    inputs = self.transform(inputs)
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
