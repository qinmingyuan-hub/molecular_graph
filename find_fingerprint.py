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

# List of dataset names
dataset_names = ["BT-20.csv", "BT-20-1.csv", "BT-474.csv", "BT-549.csv",
                 "HBL-100.csv", "HS-578T.csv", "MCF-7.csv", "MDA-MB-231.csv",
                 "MDA-MB-361.csv", "MDA-MB-435.csv", "MDA-MB-453.csv",
                 "MDA-MB-468.csv", "SK-BR-3.csv", "T-47D.csv"]

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Function to generate and save fingerprints
def generate_and_save_fingerprints(smiles, fingerprint_functions, fingerprint_dir, dataset_name):
    for func in tqdm(fingerprint_functions, desc="Generating fingerprints"):
        fingerprint = func(smiles)  # Generate fingerprint
        fingerprint_path = os.path.join(fingerprint_dir, f"{func.__name__}.{dataset_name.split('.')[0]}.npy")
        np.save(fingerprint_path, fingerprint)  # Save fingerprint
        # Validate fingerprint shape
        print(f"Generated fingerprint {func.__name__}: {fingerprint.shape} (samples: {fingerprint.shape[0]}, features: {fingerprint.shape[1]})")
    print("Fingerprints saved successfully.")


# Function to train the model
def train_model(features, labels, combination_name):
    # Validate feature and label shapes
    if features.shape[0] != labels.shape[0]:
        print(f"Error! Feature sample count ({features.shape[0]}) does not match label sample count ({labels.shape[0]})!")
        return 0, 0

    # Create a mask for valid samples - skip samples where all labels are NaN
    valid_mask = ~np.all(np.isnan(labels), axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]

    print(f"Valid samples after filtering: {features.shape[0]} (original: {valid_mask.shape[0]})")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Print the shapes of the split data
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    # Build the model
    model = SimpleNet(X_train.shape[1], y_train.shape[1])
    print(f"Model input size: {X_train.shape[1]}, output size: {y_train.shape[1]}")

    # Define loss function and optimizer
    criterion = nn.BCELoss(reduction='none')  # Set to none for later processing
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr = 0.001  # Learning rate

    # Use DataLoader to load data in batches
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize patience and best AUC
    patience = 5
    best_auc = 0
    counter = 0

    # Train the model
    for epoch in tqdm(range(30), desc=f"Training {combination_name}"):
        model.train()
        epoch_loss_sum = 0.0  # Sum of all valid losses
        total_valid_elements = 0  # Count of all valid elements

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Create a mask for valid label positions (non-NaN)
            mask = ~torch.isnan(batch_y)

            # Temporarily replace NaN with 0 (to avoid loss function errors)
            clean_batch_y = torch.where(mask, batch_y, torch.zeros_like(batch_y))

            # Calculate element-wise loss
            loss_per_element = criterion(outputs, clean_batch_y)

            # Zero out loss at invalid positions (NaN)
            masked_loss = loss_per_element * mask.float()

            # Calculate the sum of valid losses and count of valid elements in the current batch
            batch_valid_loss = masked_loss.sum()
            num_valid = mask.sum()

            if num_valid > 0:
                # Calculate batch average loss (for backpropagation)
                batch_loss = batch_valid_loss / num_valid
                batch_loss.backward()
                optimizer.step()

                # Accumulate loss and valid elements for the entire epoch
                epoch_loss_sum += batch_valid_loss.item()
                total_valid_elements += num_valid.item()
            else:
                # Skip optimization if the entire batch is NaN
                print(f"Warning: Batch {batch_X.size(0)} has all invalid labels")
                continue

        if total_valid_elements > 0:
            avg_epoch_loss = epoch_loss_sum / total_valid_elements

            # Calculate the ratio of valid samples
            total_samples = len(train_loader.dataset)  # Total number of samples in the dataset
            valid_ratio = total_valid_elements / (total_samples * y_train.shape[1]) * 100

            print(f"Epoch {epoch + 1}/{30}: Average loss = {avg_epoch_loss:.6f}, "
                  f"Valid sample ratio = {valid_ratio:.1f}%")
        else:
            avg_epoch_loss = float('nan')
            print(f"Epoch {epoch + 1}/{30}: No valid samples, skipping training")

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            predicted_probs = outputs.numpy()

            # Calculate AUC for each label
            aucs = []
            valid_labels = []  # Record indices of valid labels

            for i in range(y_val.shape[1]):
                # Get validation data for the current label
                y_true = y_val.numpy()[:, i]
                y_pred = predicted_probs[:, i]

                # Create a mask to skip NaN values
                mask = ~np.isnan(y_true)
                y_true_masked = y_true[mask]
                y_pred_masked = y_pred[mask]

                if len(y_true_masked) == 0:
                    # Skip labels with no valid samples
                    print(f"Skipping label {i} in combination {combination_name} - no valid samples")
                    continue

                # Check if the label is valid (contains at least two classes)
                if len(np.unique(y_true_masked)) > 1:
                    try:
                        auc = roc_auc_score(y_true_masked, y_pred_masked)
                        aucs.append(auc)
                        valid_labels.append(i)
                    except ValueError:
                        print(
                            f"Warning: Unable to calculate AUC for label {i} in combination {combination_name}. Using accuracy as proxy.")
                        predicted = (torch.FloatTensor(y_pred_masked) > 0.5).float()
                        auc = (predicted.numpy().flatten() == y_true_masked).mean()
                        aucs.append(auc)
                        valid_labels.append(i)
                else:
                    # Skip invalid labels (single class)
                    print(f"Skipping label {i} in combination {combination_name} - only one class present")

            # Calculate average AUC (only for valid labels)
            if aucs:
                auc = np.mean(aucs)
                print(f"Validated {len(aucs)}/{y_val.shape[1]} labels for {combination_name}, Avg AUC: {auc:.4f}")
            else:
                print(f"Warning: No valid labels for combination {combination_name}. Using 0 as proxy.")
                auc = 0

        # Check if the current AUC is the best
        if auc > best_auc:
            best_auc = auc
            counter = 0
        else:
            counter += 1

        # Check if patience has been reached
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in AUC for {patience} epochs.")
            break

    return best_auc, len(valid_labels)


# Function to process a single dataset
def process_dataset(dataset_name):
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 50}")

    # Load the data
    file_path = 'Data/BreastCellLines/' + dataset_name
    data = pd.read_csv(file_path)
    smiles = data.iloc[:, 0].tolist()
    labels = data.iloc[:, 1:].values  # Multi-label data, starting from the second column

    # Check the range of label values and NaN values
    print("Label value range:", np.nanmin(labels), "to", np.nanmax(labels))
    print("Number of NaN labels:", np.isnan(labels).sum(), f"({np.isnan(labels).mean() * 100:.2f}%)")

    # If labels are not in the range [0,1], convert them
    if np.nanmin(labels) < 0 or np.nanmax(labels) > 1:
        print("Labels not in [0,1] range, converting to binary...")
        # Convert labels to 0 or 1, while preserving NaN values
        labels = np.where(np.isnan(labels), np.nan, np.where(labels > 0.5, 1.0, 0.0))
        print("Converted label value range:", np.nanmin(labels), "to", np.nanmax(labels))

    # Generate and save fingerprints
    generate_and_save_fingerprints(smiles, fingerprint_functions, fingerprint_dir, dataset_name)

    # List to store results
    results = []
    best_score = 0
    best_combination = None
    lr = 0.001  # Learning rate

    # Generate all combinations of 1, 2, and 3 fingerprints
    for r in [1, 2, 3]:
        combinations_list = list(combinations(fingerprint_functions, r))
        for combination in tqdm(combinations_list, desc=f"Training {r}-fingerprint combinations for {dataset_name}"):
            combination_features = []
            combination_name = "+".join([func.__name__ for func in combination])

            print(f"\nProcessing combination: {combination_name}")

            # Load each fingerprint as needed
            for func in combination:
                # Ensure the correct file name format
                fingerprint_path = os.path.join(fingerprint_dir, f"{func.__name__}.{dataset_name.split('.')[0]}.npy")
                print(f"Loading fingerprint: {fingerprint_path}")

                if not os.path.exists(fingerprint_path):
                    print(f"Error! Fingerprint file does not exist: {fingerprint_path}")
                    continue

                fingerprint = np.load(fingerprint_path)
                print(f"Fingerprint {func.__name__} shape: {fingerprint.shape}")

                # Check if the number of fingerprint samples matches the number of SMILES
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

            # Ensure the number of feature samples matches the number of label samples
            if features.shape[0] != labels.shape[0]:
                print(f"Critical error! Feature sample count ({features.shape[0]}) does not match label sample count ({labels.shape[0]})!")
                continue

            # Train the model and get evaluation results
            auc, valid_labels = train_model(features, labels, combination_name)

            # Save the result data
            save_data = {
                'dataset': dataset_name,
                'combination': combination_name,
                'auc': auc,
                'valid_labels': valid_labels,
                'total_labels': labels.shape[1],
                'lr': lr,
            }
            print(save_data)
            results.append(save_data)

            # Free memory
            del combination_features, features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Update the best combination
            if auc > best_score:
                best_score = auc
                best_combination = combination_name

    # Save the results to a CSV file
    if results:
        results_df = pd.DataFrame(results)
        # Include the dataset name in the file name
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

    # Optionally: Save all dataset results to a summary file
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv('all_fingerprint_combination_results.csv', index=False)
        print("All dataset results saved to all_fingerprint_combination_results.csv")

if __name__ == "__main__":
    main()