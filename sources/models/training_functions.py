import torch.nn as nn
import torch
import torchmetrics
from models.testing_functions import eval
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List

def create_writer(experiment_name: str,
                  model_name: str,
                  writer_dir: str,
                  extra: str = None):
    
    from datetime import datetime
    import os

    # Get timestamp of current date in reverse order
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Create log dir path
    if extra:
        log_dir = os.path.join(writer_dir, timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(writer_dir, timestamp, experiment_name, model_name)

    print(f"[INFO] Created summary writer was saved to {log_dir}")

    return SummaryWriter(log_dir = log_dir)


def train(model: nn.Module,
          training_dataloader: torch.utils.data.DataLoader,
          testing_dataloader: torch.utils.data.DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          metric_fn: torchmetrics,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:


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

        # Put results in a SummaryWriter for tracking experiment
        if writer:

            # Add loss results to SummaryWriter
            writer.add_scalars(main_tag = "Loss",
                               tag_scalar_dict = {"train_loss": train_loss_value,
                                                  "test_loss": test_loss_value},
                               global_step = epoch
                               )
            
            # Add metric results to SummaryWriter
            writer.add_scalars(main_tag = "Metric",
                               tag_scalar_dict = {"train_metric": train_metric_value,
                                                  "test_metric": test_metric_value},
                               global_step = epoch)

            # Close the writer
            writer.close

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

