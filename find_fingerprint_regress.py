import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from itertools import combinations
from model.data.scikit_fingerprint import (
    get_PubChemFingerprint, get_MACCSFingerprint, get_ERGFingerprint,
    get_ECFPFingerprint, get_RDKitFingerprint, get_AvalonFingerprint,
    get_EStateFingerprint, get_GhoseCrippenFingerprint
)
import torch.nn.functional as F

# List of dataset names
dataset_names = ["esol.csv", "freesolv.csv", "HLM_CLint.csv", "HUMAN.csv",
                 "lipo.csv", "MDR1-MDCK.csv", "RAT.csv", "RLM_CLint.csv",
                 "SOLUBILITY.csv"]

# List of fingerprint functions
fingerprint_functions = [
    get_PubChemFingerprint,
    get_MACCSFingerprint,
    get_ERGFingerprint,
    get_ECFPFingerprint,
    get_RDKitFingerprint,
    get_AvalonFingerprint,
    get_EStateFingerprint,
    get_GhoseCrippenFingerprint
]

# Directory to save fingerprints
fingerprint_dir = "fingerprints"
os.makedirs(fingerprint_dir, exist_ok=True)


# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Generate and save fingerprints
def generate_and_save_fingerprints(smiles, fingerprint_functions, fingerprint_dir, dataset_name):
    for func in tqdm(fingerprint_functions, desc="Generating fingerprints"):
        fingerprint = func(smiles)  # Generate fingerprint
        fingerprint_path = os.path.join(fingerprint_dir, f"{func.__name__}.{dataset_name.split('.')[0]}.npy")
        np.save(fingerprint_path, fingerprint)  # Save fingerprint
        # Validate fingerprint shape
        print(f"Generated fingerprint {func.__name__}: {fingerprint.shape} (Samples: {fingerprint.shape[0]}, Features: {fingerprint.shape[1]})")
    print("Fingerprints saved successfully.")

# Train the model
def train_model(features, labels, combination_name):
    # Validate feature and label shapes
    if features.shape[0] != labels.shape[0]:
        print(f"Error! Feature samples ({features.shape[0]}) do not match label samples ({labels.shape[0]})!")
        return float('inf'), 0  # Return infinite loss and 0 valid labels

    # Create a mask for valid samples - skip samples where all labels are NaN
    valid_mask = ~np.all(np.isnan(labels), axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]

    print(f"Filtered valid samples: {features.shape[0]} (Original: {valid_mask.shape[0]})")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Print shapes after splitting
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    # Build the regression model (assuming SimpleNet is modified for regression)
    model = SimpleNet(X_train.shape[1], y_train.shape[1])
    print(f"Model input size: {X_train.shape[1]}, Output size: {y_train.shape[1]}")

    # Define loss function and optimizer - Using MSELoss
    criterion = nn.MSELoss(reduction='none')  # Set to 'none' for later processing
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Use DataLoader for batch loading
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize patience and best validation loss
    patience = 5
    best_val_loss = float('inf')  # Initially set to infinity
    counter = 0

    # Train the model
    for epoch in tqdm(range(60), desc=f"Training {combination_name}"):
        model.train()
        epoch_loss_sum = 0.0
        total_valid_elements = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Create a mask for valid labels (non-NaN)
            mask = ~torch.isnan(batch_y)

            # Temporarily replace NaN with 0 (to avoid loss function errors)
            clean_batch_y = torch.where(mask, batch_y, torch.zeros_like(batch_y))

            # Calculate element-wise loss
            loss_per_element = criterion(outputs, clean_batch_y)

            # Zero out loss for invalid positions (NaN)
            masked_loss = loss_per_element * mask.float()

            # Calculate total valid loss and number of valid elements for the batch
            batch_valid_loss = masked_loss.sum()
            num_valid = mask.sum()

            if num_valid > 0:
                # Calculate batch average loss (for backpropagation)
                batch_loss = batch_valid_loss / num_valid
                batch_loss.backward()
                optimizer.step()

                # Accumulate epoch loss and valid elements
                epoch_loss_sum += batch_valid_loss.item()
                total_valid_elements += num_valid.item()
            else:
                print(f"Warning: Batch {batch_X.size(0)} has all invalid labels")
                continue

        if total_valid_elements > 0:
            avg_epoch_loss = epoch_loss_sum / total_valid_elements
            print(f"Epoch {epoch + 1}/{60}: Training loss = {avg_epoch_loss:.6f}")
        else:
            avg_epoch_loss = float('nan')
            print(f"Epoch {epoch + 1}/{60}: No valid samples, skipping training")

        # Validation phase
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss_per_element = criterion(outputs, y_val)

            # Create validation mask
            val_mask = ~torch.isnan(y_val)

            # Calculate validation loss (only for valid elements)
            val_loss_masked = val_loss_per_element * val_mask.float()
            total_val_elements = val_mask.sum().item()

            if total_val_elements > 0:
                avg_val_loss = val_loss_masked.sum().item() / total_val_elements
                print(f"Epoch {epoch + 1}/{60}: Validation loss = {avg_val_loss:.6f}")

                # Calculate RMSE for each label
                rmses = []
                for i in range(y_val.shape[1]):
                    # Get validation data for the current label
                    y_true = y_val[:, i]
                    y_pred = outputs[:, i]

                    # Create mask to skip NaN values
                    mask = ~torch.isnan(y_true)
                    y_true_masked = y_true[mask]
                    y_pred_masked = y_pred[mask]

                    if len(y_true_masked) > 0:
                        mse = F.mse_loss(y_pred_masked, y_true_masked).item()
                        rmse = np.sqrt(mse)
                        rmses.append(rmse)

                # Calculate average RMSE
                if rmses:
                    avg_rmse = np.mean(rmses)
                    print(f"Validation set average RMSE: {avg_rmse:.4f}")
            else:
                avg_val_loss = float('inf')
                print("Validation set has no valid labels, setting validation loss to infinity")

        # Early stopping check: Monitor validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1

        # Check patience
        if counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.")
            break

    return best_val_loss, total_val_elements


# Process a single dataset
def process_dataset(dataset_name):
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 50}")

    # Load data
    file_path = 'Data/regress/' + dataset_name
    data = pd.read_csv(file_path)
    smiles = data.iloc[:, 0].tolist()
    labels = data.iloc[:, 1:].values  # Multi-label data, starting from the second column

    # Check label value range and NaN values
    print("Label value range:", np.nanmin(labels), "to", np.nanmax(labels))
    print("NaN label count:", np.isnan(labels).sum(), f"({np.isnan(labels).mean() * 100:.2f}%)")

    # Generate and save fingerprints
    generate_and_save_fingerprints(smiles, fingerprint_functions, fingerprint_dir, dataset_name)

    # List to store results
    results = []
    best_score = 999
    best_combination = None
    lr = 0.001  # Learning rate

    # Generate all 1, 2, 3 fingerprint combinations
    for r in [1, 2, 3]:
        combinations_list = list(combinations(fingerprint_functions, r))
        for combination in tqdm(combinations_list, desc=f"Training {r}-fingerprint combinations for {dataset_name}"):
            combination_features = []
            combination_name = str([func.__name__.replace('get_', '') for func in combination])

            print(f"\nProcessing combination: {combination_name}")

            # Load each fingerprint as needed
            for func in combination:
                # Ensure correct filename format
                fingerprint_path = os.path.join(fingerprint_dir, f"{func.__name__}.{dataset_name.split('.')[0]}.npy")
                print(f"Loading fingerprint: {fingerprint_path}")

                if not os.path.exists(fingerprint_path):
                    print(f"Error! Fingerprint file does not exist: {fingerprint_path}")
                    continue

                fingerprint = np.load(fingerprint_path)
                print(f"Fingerprint {func.__name__} shape: {fingerprint.shape}")

                # Check if fingerprint sample count matches
                if fingerprint.shape[0] != len(smiles):
                    print(f"Warning! Fingerprint {func.__name__} sample count ({fingerprint.shape[0]}) does not match SMILES count ({len(smiles)})")

                combination_features.append(fingerprint)

            # Skip this combination if no features were loaded
            if not combination_features:
                print(f"Error! Could not load any features for combination {combination_name}")
                continue

            # Concatenate features
            features = np.hstack(combination_features)
            print(f"Combined feature shape: {features.shape}")

            # Ensure feature sample count matches label sample count
            if features.shape[0] != labels.shape[0]:
                print(f"Critical error! Feature sample count ({features.shape[0]}) does not match label sample count ({labels.shape[0]})!")
                continue

            # Train the model and get evaluation results
            RMSE, valid_labels = train_model(features, labels, combination_name)

            # Save result data
            save_data = {
                'dataset': dataset_name,
                'combination': combination_name,
                'RMSE': RMSE,
                'valid_labels': valid_labels,
                'total_labels': labels.shape[1],
                'lr': lr,
            }
            print(save_data)
            results.append(save_data)

            # Free memory
            del combination_features, features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Update best combination
            if RMSE < best_score:
                best_score = RMSE
                best_combination = combination_name

    # Save results to a CSV file
    if results:
        results_df = pd.DataFrame(results)
        # Include dataset name in the filename
        results_filename = f'fingerprint_combination_results_{dataset_name.split(".")[0]}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"Results for {dataset_name} saved to {results_filename}")

        print(f"\nBest combination for dataset {dataset_name}: {best_combination}")
        print(f"Best AUC: {best_score}")
    else:
        print(f"Error! No results generated for dataset {dataset_name}")

    return results


# Main function
def main():
    all_results = []

    for dataset_name in dataset_names:
        dataset_results = process_dataset(dataset_name)
        all_results.extend(dataset_results)

    # Optional: Save all dataset results to a summary file
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv('all_fingerprint_combination_results.csv', index=False)
        print("Results for all datasets saved to all_fingerprint_combination_results.csv")

if __name__ == "__main__":
    main()