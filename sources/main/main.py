# -data_dir = "C:\Users\axell\Documents\dev\cnn\data"
# python main.py -data_dir "C:\Users\axell\Documents\dev\cnn\data"

import numpy as np
from pathlib import Path
import argparse
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from models.CNN import CNN
from models.training_functions import train
from models.testing_functions import eval

def main(data_dir: str):

    DATA_DIR = Path(data_dir)

    # Load the data 
    training_data = datasets.FashionMNIST(
            root = DATA_DIR,
            train = True,
            download = True,
            transform = ToTensor()
            )

    testing_data = datasets.FashionMNIST(
            root = DATA_DIR,
            train = False,
            download = True,
            transform = ToTensor()
            )
    
    # Put the data inside dataloaders
    BATCH_SIZE = 32
    
    class_names = training_data.classes
    num_classes = int(len(class_names))
    print(num_classes)

    training_dataloader = DataLoader(dataset = training_data,
                                     batch_size = BATCH_SIZE,
                                     shuffle = True)

    testing_dataloader = DataLoader(dataset = testing_data,
                                    batch_size = BATCH_SIZE,
                                    shuffle = False)

    # Initialize the model
    model = CNN(in_channels = 1,
                out_channels = num_classes)

    # Initialize loss, optimizer, and metric functions
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(params = model.parameters(),
                     lr = 1e-3)
    metric_fn = Accuracy(task = "multiclass", num_classes = num_classes)

    # Device agnostic code
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Train the network
    EPOCHS = 10

    train(model = model,
          dataloader = training_dataloader,
          loss_fn = loss_fn,
          optimizer = optimizer,
          metric_fn = metric_fn,
          epochs = EPOCHS,
          device = device)

    # Evaluate the model
    eval(model = model,
         dataloader = testing_dataloader,
         loss_fn = loss_fn,
         metric_fn = metric_fn,
         device = device)

    return

if __name__ == "__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required = True, help = "Path toward data directory")
    args = parser.parse_args()

    main(data_dir = args.data_dir)


