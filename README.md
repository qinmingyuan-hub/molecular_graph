# How to Use
## 1. Configure Parameters
You can configure training parameters in two ways:

### a) Modify Directly in the Code (Recommended for Quick Experiments)
Open the `train_main.py` file and locate the `if __name__ == '__main__':` code block. You can directly modify the attributes of the `args` object here:

```python
# train_main.py

if __name__ == '__main__':
    args = set_train_argument()
    # ...

    # --- Modify parameters here ---
    args.num_folds = 10              # Number of cross-validation folds
    args.epochs = 20                 # Number of training epochs
    args.dataset_type = 'regression' # Task type: 'regression' or 'classification'
    args.model_type = 'moleculeformer_mini' # Model architecture type
    args.metric = 'rmse'             # Evaluation metric
    args.split_type = 'random'       # Dataset splitting method
    args.dropout = 0                 # Dropout rate
    args.hidden_size = 100           # Model hidden layer size
    args.seed = 0                    # Random seed (ensures reproducibility)
    
    # --- Specify file paths here ---
    path = 'LOG HLM_CLint (mL_min_kg).csv'  # Dataset file name
    args.data_path = 'Data/ADME/split_results/' + path # Full dataset path
    # ...
```

#### Key Parameter Explanations:
- **dataset_type**: Specifies whether the task is regression or classification.
- **model_type**: Selects the model architecture to use.
- **metric**: Selects the metric for evaluating model performance:
  - For regression tasks: `rmse`, `mse`, `mae`, `r2`, etc.
  - For classification tasks: `auc`, `prc-auc`, `accuracy`, etc.
- **split_type**: Defines how to split the dataset (e.g., `random` for random splitting).
- **seed**: Sets a random seed to ensure consistent results for data splitting and model initialization each time you run the same code.


### b) Pass via Command-Line Arguments
The `set_train_argument()` function allows you to override default parameters via the command line. Example:

```bash
python train_main.py --epochs 50 --model_type moleculeformer --seed 42
```


## 2. Prepare Data
Ensure your dataset file (e.g., `LOG HLM_CLint (mL_min_kg).csv`) is placed in the directory specified by `args.data_path`.

The CSV file must typically include the following columns:
- **smiles**: SMILES string representation of the molecule.
- **label**: Target property value of the molecule (numeric values for regression tasks, 0/1 for classification tasks).


## 3. Run Training
You can run the training script in two ways:

### a) Local Run (For Debugging)
Run directly in the terminal for easy output viewing and debugging:

```bash
python train_main.py
```


### b) Background Run (Recommended for Long-Term Training)
For long training sessions, use the `nohup` command to run the script in the background (training continues even if you close the terminal):

```bash
nohup python train_main.py > training_log.log 2>&1 &
```

- `nohup ... &`: Runs the command continuously in the background.
- `> training_log.log`: Redirects standard output (stdout) to the `training_log.log` file.
- `2>&1`: Redirects standard error (stderr, e.g., error messages) to standard output, so errors are also written to `training_log.log`.


## 4. View Results
- **Log Files**: Logs of the training process (including loss and evaluation metrics for each epoch) are saved to the file specified by `args.log_path`. The filename is automatically generated based on your parameters (e.g., `log_moleculeformer_mini_random_rmse_0_NO.log`).
- **Model Files**: Trained model weights are saved to the directory specified by `args.save_path`. The directory name is also automatically generated (e.g., `model_save_moleculeformer_mini_random_rmse_0_NO`).


## Notes
- **CUDA Availability**: The code automatically detects and uses available GPUs (if you have a CUDA-enabled version of PyTorch installed). It will use the CPU if no GPU is available.
- **Data Path**: Ensure `args.data_path` points to the correct dataset file.
- **Parameter Consistency**: The selected `args.metric` must match `args.dataset_type`. For example, `auc` is not suitable for regression tasks.
- **Log & Model Paths**: `save_path` and `log_path` automatically include key parameters to help you distinguish and manage results from different experiments.