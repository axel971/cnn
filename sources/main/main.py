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
from models.training_functions import train, create_writer
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

    # Device agnostic code
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = CNN(in_channels = 1,
                out_channels = num_classes).to(device)
    model.name = "cnn"

    # Initialize loss, optimizer, and metric functions
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(params = model.parameters(),
                     lr = 1e-3)
    metric_fn = Accuracy(task = "multiclass", num_classes = num_classes)

    # Train the network
    EPOCHS = 10

    train(model = model,
          training_dataloader = training_dataloader,
          testing_dataloader = testing_dataloader,
          loss_fn = loss_fn,
          optimizer = optimizer,
          metric_fn = metric_fn,
          epochs = EPOCHS,
          device = device,
          writer = create_writer(experiment_name =f"{EPOCHS}_epochs",
                                 model_name = model.name,
                                 writer_dir = DATA_DIR / "logs" )
          )

    # Save the model
    SAVE_MODEL_DIR = DATA_DIR /"models"/ model.name / "weights" 
    SAVE_MODEL_DIR.mkdir(parents = True, exist_ok = True)

    torch.save(model.state_dict(), SAVE_MODEL_DIR / "final_weights.pth")

    return

if __name__ == "__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("-data_dir", required = True, help = "Path toward data directory")
    args = parser.parse_args()

    main(data_dir = args.data_dir)


