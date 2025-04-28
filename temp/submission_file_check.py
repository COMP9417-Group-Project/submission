import numpy as np
import pandas as pd
import zipfile


# open zip file containing preds_1.npy and preds_2.npy
with zipfile.ZipFile("Can't think of a name but still here.zip", 'r') as zip_ref: 
    zip_ref.extractall('extracted_files') # Extract all files into the 'extracted_files'folder

preds_1 = np.load('extracted_files/preds_1.npy') 
preds_2 = np.load('extracted_files/preds_2.npy')
# Check if preds_1 is of size 1000x28 and preds_2 is of size 1818x28
if preds_1.shape != (1000, 28):
    raise ValueError(f"preds_1 has size {preds_1.shape}, but expected 1000x28") 
if preds_2.shape != (1818, 28):
    raise ValueError(f"preds_2 has size {preds_2.shape}, but expected 1818x28")
def weighted_log_loss(y_true, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.
    Parameters:
    - y_true: (N, C) One-hot encoded true labels - y_pred: (N, C) Predicted probabilities
    Returns:
    - Weighted log loss (scalar).
    """
    # Compute class frequencies
    class_counts = np.sum(y_true, axis=0) # Sum over samples to get counts per class 
    class_weights = 1.0 / class_counts
    class_weights /= np.sum(class_weights) # Normalize weights to sum to 1
    # Compute weighted loss
    sample_weights = np.sum(y_true * class_weights, axis=1) # Get weight for each sample 
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))
    return loss

    # y_test_1_ohe is the one hot encoded array of true labels in test set 1
    # y_test_2_ohe is the one hot encoded array of true labels in test set 2
    # you do not have access to either, here are RANDOMLY generated ohe labels to ensure code runs
    
y_test_1_ohe = (np.arange(28) == np.random.choice(28, size=1000)[:, None]).astype(int) 
y_test_2_ohe = (np.arange(28) == np.random.choice(28, size=1818)[:, None]).astype(int)
loss_1 = weighted_log_loss(y_test_1_ohe , preds_1) 
loss_2 = weighted_log_loss(y_test_2_ohe , preds_2)

print(f"Loss 1: {loss_1}")
print(f"Loss 2: {loss_2}")