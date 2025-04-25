# train RNN on BTC time series data
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from argparse import ArgumentParser

# import local modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) # this appends 'src' to the path
from data.dataset import DatasetMaker, BaseDataset
from models.rnn import RNNModel
from logger.logger import Logger

def main(args):
    # Initialize the logger
    logger = Logger(save_dir=args.log_dir, run_name=args.run_name, args=args)

    # Load the dataset
    dataset_maker = DatasetMaker(data_path=args.data_path, val_days=args.val_days, test_days=args.test_days, seq_length=args.seq_len)

    # Create data loaders for training, validation, and testing
    dataloaders = dataset_maker.get_dataloaders()
    train_data = dataset_maker.train_data
    val_data = dataset_maker.val_data
    test_data = dataset_maker.test_data
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Initialize the model
    model = RNNModel(input_size=train_data[0][0].shape[0],  # Input size: number of features in the data
                     hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers,
                     output_size=1).to(args.device)  # Assuming 1 output (e.g., the open price for the next day)

    # Load pre-trained model if specified
    if args.load_model and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print("Pre-trained model loaded.")

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Start training
    best_val_loss = float('inf')
    train_metrics = []
    val_metrics = []

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        # Store metrics for logging
        train_metrics.append({'epoch': epoch + 1, 'train_loss': avg_train_loss})
        val_metrics.append({'epoch': epoch + 1, 'val_loss': avg_val_loss})

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.save_model:
                torch.save(model.state_dict(), args.model_path)
                print(f"Model saved at epoch {epoch+1}")

    # Log metrics to CSV files
    logger.log_df(pd.DataFrame(train_metrics), "train_metrics.csv")
    logger.log_df(pd.DataFrame(val_metrics), "val_metrics.csv")

    # Optionally, you can test the model here
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    logger.log_df(pd.DataFrame([{'test_loss': avg_test_loss}]), "test_metrics.csv")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train RNN on BTC time series data")
    parser.add_argument("--data_path", type=str, default="data/market_data_V1.csv", help="Path to the BTC data file")
    parser.add_argument("--seq_len", type=int, default=30, help="Length of the input sequence")
    parser.add_argument("--val_days", type=int, default=30, help="Number of days for validation")
    parser.add_argument("--test_days", type=int, default=30, help="Number of days for testing")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for the RNN model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers for the RNN model")
    parser.add_argument("--model_path", type=str, default="models/rnn_model.pth", help="Path to save the trained model")
    parser.add_argument("--load_model", action="store_true", help="Load a pre-trained model")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run for logging")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")

    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)