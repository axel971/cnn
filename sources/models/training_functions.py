import torch.nn as nn
import torch
import torchmetrics
from models.testing_functions import eval
from tqdm import tqdm

def train(model: nn.Module,
          training_dataloader: torch.utils.data.DataLoader,
          testing_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          metric_fn: torchmetrics,
          epochs: int,
          device):


    results = { "train_loss": [],
               "train_metric": [],
               "test_loss": [],
               "test_metric": []
               }

    for epoch in range(epochs):
        
        print(f"Epoch {epoch + 1}/{epochs}:")

        train_loss_value, train_metric_value = train_step(model = model,
                   dataloader = training_dataloader,
                   loss_fn = loss_fn,
                   optimizer = optimizer,
                   metric_fn = metric_fn,
                   device = device)

        test_loss_value, test_metric_value = eval(model = model,
                                                  dataloader = testing_dataloader,
                                                  loss_fn = loss_fn,
                                                  metric_fn = metric_fn,
                                                  device = device)

        print(f"Epoch {epoch + 1} | Train loss: {train_loss_value} | Train metric: {train_metric_value} | Test loss: {test_loss_value} | Test metric: {test_metric_value}")

        results["train_loss"].append(train_loss_value)
        results["train_metric"].append(train_metric_value)
        results["test_loss"].append(test_loss_value)
        results["test_metric"].append(test_metric_value)

    return results


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               metric_fn: torchmetrics,
               device):
    
    '''
    Function that perform training of the model per batch
    '''

    loss_value = 0

    loop = tqdm(dataloader)

    model.train() # Put the model in training mode (=> adapt batch_norm, dropout, etc)

    for (x, y) in loop:
        
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        # compute loss of the current batch
        loss = loss_fn(y_pred, y)
        loss_value += loss.item()

        # Compute metric of the current batch
        metric = metric_fn(y_pred, y)

        # Display loss and metric for the current batch
        loop.set_postfix(loss = loss.item(), metric = metric.item())

        # Update the model weights
        optimizer.zero_grad()

        loss.backward() # Perform backpropagation

        optimizer.step()
        

    loss_value = loss_value/len(dataloader)
    metric_value = metric_fn.compute()

    metric_fn.reset()

    return loss_value, metric_value

