# train RNN on BTC time series data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fixes a bug with plt and OpenMP in PyTorch
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# import local modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) # this appends 'src' to the path
from data.dataset import DatasetMaker, BaseDataset
from models.rnn import RNNModel
from models.mlp import MLPModel
from models.lstm import LSTMModel
from logger.logger import Logger

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

MODELS = {
    "rnn": RNNModel,
    "lstm": LSTMModel,
    "mlp": MLPModel
}

def main(args):
    # Set all seeds for maximal reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize the logger
    logger = Logger(save_dir=args.log_dir, run_name=args.run_name, args=args)

    # Load the dataset
    dataset_maker = DatasetMaker(data_path=args.data_path, val_days=args.val_days, test_days=args.test_days, seq_len=args.seq_len, inited = not args.process_data)

    # Create data loaders for training, validation, and testing
    dataloaders = dataset_maker.get_dataloaders()
    train_data = dataset_maker.train_data
    val_data = dataset_maker.val_data
    test_data = dataset_maker.test_data
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # Initialize the model
    if args.model_type not in MODELS:
        raise ValueError(f"Model type {args.model_type} not recognized. Choose from {list(MODELS.keys())}.")
    else:
        model_class = MODELS[args.model_type]
        model = model_class(input_dim=train_data[0][0].shape[-1], hidden_dim=args.hidden_dim, seq_len=args.seq_len, output_dim=1, num_layers=args.num_layers, dropout=args.dropout).to(args.device)

    # Load pre-trained model
    model_path = os.path.join(args.log_dir, args.run_name, "model.pth")
    if args.load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Training
    best_val_loss = float('inf')
    train_metrics = []
    val_metrics = []

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training", ncols=100)):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / (len(train_loader) * args.batch_size)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation", ncols=100):
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / (len(val_loader) * args.batch_size)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        # scheduler.step(avg_val_loss)

        # Store metrics for logging
        train_metrics.append({'epoch': epoch + 1, 'train_loss': avg_train_loss})
        val_metrics.append({'epoch': epoch + 1, 'val_loss': avg_val_loss})

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.save_model:
                model_path = os.path.join(args.log_dir, args.run_name, "model.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved at epoch {epoch+1}")

        # update the learning rate
        

    # Log metrics to CSV files
    logger.log_df(pd.DataFrame(train_metrics), "train_metrics.csv")
    logger.log_df(pd.DataFrame(val_metrics), "val_metrics.csv")

    # Testing
    model.eval()
    test_loss = 0.0
    
    # Test the model on the test set and save outputs for analysis
    outputs = {"predictions": [], "actuals": []}
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing", ncols=100):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            output = model(x_batch)

            loss = criterion(output, y_batch)
            test_loss += loss.item()
            outputs["predictions"].append(output.cpu().numpy())
            outputs["actuals"].append(y_batch.cpu().numpy())
        # Save predictions and actuals to CSV for analysis
        outputs_df = pd.DataFrame({"predictions": np.concatenate(outputs["predictions"]), "actuals": np.concatenate(outputs["actuals"])})
        logger.log_df(outputs_df, "test_outputs.csv")
        print("Test outputs saved to test_outputs.csv")


    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    logger.log_df(pd.DataFrame([{'test_loss': avg_test_loss}]), "test_metrics.csv")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot([m['epoch'] for m in train_metrics], [m['train_loss'] for m in train_metrics], label='Train Loss')
    plt.plot([m['epoch'] for m in val_metrics], [m['val_loss'] for m in val_metrics], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(logger.run_dir, 'loss_plot.png'))
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(description="Train RNN on BTC time series data")
    parser.add_argument("--data_path", type=str, default="../data/market_data_V1.csv", help="Path to the BTC data file")
    parser.add_argument("--seq_len", type=int, default=30, help="Length of the input sequence")
    parser.add_argument("--val_days", type=int, default=30, help="Number of days for validation")
    parser.add_argument("--test_days", type=int, default=30, help="Number of days for testing")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for the RNN model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers for the RNN model")
    parser.add_argument("--load_model", action="store_true", help="Load a pre-trained model")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--process_data", action="store_true", help="Process data before training")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run for logging")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for the RNN model" )
    parser.add_argument("--model_type", type=str, choices=MODELS.keys(), default="rnn", help="Type of model to use (rnn, lstm, mlp)")

    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)