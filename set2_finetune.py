# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score 
from sklearn.utils.class_weight import compute_class_weight
import logging
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import random

import sys

# define path
y_train_path = 'y_train.csv'
y_test_path = 'y_test_2_reduced.csv'
x_train_path = 'X_train.csv'
x_test_one_path = 'X_test_1.csv'
x_test_two_path = 'X_test_2.csv'


# read df
y = pd.read_csv(y_train_path)
y_test_two_label = pd.read_csv(y_test_path)
X_train_df = pd.read_csv(x_train_path)
X_test_one = pd.read_csv(x_test_one_path)
X_test_two = pd.read_csv(x_test_two_path)

test_length = y_test_two_label.shape[0]
test_indices = np.arange(test_length)
X_test_two_labelled = X_test_two.iloc[test_indices]
mask = np.ones(len(X_test_two), dtype=bool)
mask[test_indices] = False
X_test_two_unlabelled = X_test_two.iloc[mask]

# print(X_test_two_labelled.shape)
# print(X_test_two_unlabelled.shape)



# define scaler
scaler = MinMaxScaler()
scaler.fit(X_train_df)

# transform data
X_train_scaled = scaler.transform(X_train_df)
X_val = scaler.transform(X_test_two_labelled)
X_test = scaler.transform(X_test_two_unlabelled)
y_val = y_test_two_label


# based on the y label frequency, sort the the label 
labels = y.values.flatten()
label_frequency = Counter(labels)
sorted_labels = [label for label, _ in label_frequency.most_common()]
# print(label_frequency)
sorted_labels.reverse()
num_classes = len(sorted_labels)

    

def weighted_log_loss_test(y_true, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.
    
    Parameters:
    - y_true: pandas DataFrame containing true labels (range 0-27)
    - y_pred: list containing predicted labels (range 0-27)
    
    Returns:
    - Weighted log loss (scalar).
    """
    import numpy as np
    import pandas as pd
    
    y_true_np = y_true.values.flatten() 
    y_pred_np = np.array(y_pred)
    
    # # Print the first few values for debugging
    # print("First few true labels:", y_true_np[:5])
    # print("First few predicted labels:", y_pred_np[:5])
    
    # Number of classes
    n_classes = 28  # Classes 0-27
    
    # Convert discrete labels to one-hot encoded format
    def to_one_hot(labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            if 0 <= label < num_classes:  # Safety check
                one_hot[i, label] = 1
        return one_hot
    
    # Convert true labels to one-hot format
    y_true_one_hot = to_one_hot(y_true_np, n_classes)
    
    # For predicted labels, we need probabilities
    y_pred_proba = np.zeros((len(y_pred_np), n_classes))
    for i, pred in enumerate(y_pred_np):
        if 0 <= pred < n_classes:  # Safety check
            # Assign high probability to predicted class, distribute rest among other classes
            y_pred_proba[i, :] = 0.1 / (n_classes - 1)  # Small probability for all classes
            y_pred_proba[i, pred] = 0.9  # High probability for predicted class
        else:
            # Handle invalid prediction by using uniform distribution
            y_pred_proba[i, :] = 1.0 / n_classes
    
    # Ensure no zeros in probabilities (to avoid log(0))
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1.0)
    
    # Normalize probabilities to sum to 1 for each sample
    row_sums = y_pred_proba.sum(axis=1, keepdims=True)
    y_pred_proba = y_pred_proba / row_sums
    
    # Compute class frequencies
    class_counts = np.sum(y_true_one_hot, axis=0)  # Sum over samples to get counts per class
    
    # Compute class weights with safety check for zero counts
    class_weights = np.zeros_like(class_counts)
    for c in range(n_classes):
        if class_counts[c] > 0:  # Avoid division by zero
            class_weights[c] = 1.0 / class_counts[c]
    
    # Normalize weights to sum to 1
    class_weights /= np.sum(class_weights)
    
    # Compute weighted loss
    sample_weights = np.sum(y_true_one_hot * class_weights, axis=1)  # Get weight for each sample
    
    # Calculate log loss term
    log_terms = np.sum(y_true_one_hot * np.log(y_pred_proba), axis=1)
    loss = -np.mean(sample_weights * log_terms)
    
    return loss


# ### Check the class distribution between training and test two labelled data



def plot_class_distribution(source_y, target_y, title):
    source_y = source_y.values.flatten()
    target_y = target_y.values.flatten()
    unique_classes = np.unique(np.concatenate([source_y, target_y]))
    source_counts = [np.sum(source_y == c) for c in unique_classes]
    target_counts = [np.sum(target_y == c) for c in unique_classes]

    plt.figure()
    plt.bar(unique_classes, source_counts, alpha=0.5, label='Train')
    plt.bar(unique_classes, target_counts, alpha=0.5, label='Test Two Labelled')
    plt.title(title)
    plt.legend()

plot_class_distribution(y, y_test_two_label, "Class Distribution: Source vs Test Two Labelled")


# From the above result, we can see there is the sign of piror shift where $P(Y)$ changes, e.g. the probability of the class being 5 in the training set, is far more higher than it is in the test set 2.

# ### Compare the effect of adaptive weights on the same model



def class_conditional_weights(source_X, source_y, target_X, target_y):
    """
    Calculate class-conditional importance weights for domain adaptation.
    """
    print(type(source_X))
    source_y_np = np.array(source_y).flatten()
    
    target_y_np = np.array(target_y).flatten()
        
    # Get unique classes
    unique_classes = np.unique(source_y_np)
    weights = np.ones(len(source_X))  # Initialize with ones
    
    for cls in unique_classes:
        # Get source/target indices for this class
        source_indices = np.where(source_y_np == cls)[0]
        target_indices = np.where(target_y_np == cls)[0]
        
        # Check if class is present in target
        if len(target_indices) == 0:
            # If class missing in target, set weight=0
            weights[source_indices] = 0
            continue
        
        # Extract source and target samples for this class

        source_X_cls = source_X[source_indices]
            
        target_X_cls = target_X[target_indices]
            
        # Compute class-specific domain weights
        X_domain = np.vstack([source_X_cls, target_X_cls])
        y_domain = np.concatenate([np.zeros(len(source_indices)), np.ones(len(target_indices))])
        
        try:
            lr = LogisticRegression(class_weight='balanced', max_iter=1000)
            lr.fit(X_domain, y_domain)
            
            # Get probabilities for source samples
            probs = lr.predict_proba(source_X_cls)
            
            # Add small constant to avoid division by zero
            epsilon = 1e-10
            cls_weights = (probs[:, 1] + epsilon) / (probs[:, 0] + epsilon)
            
            # Clip and normalize weights for stability
            cls_weights = np.clip(cls_weights, 0.1, 10)
            cls_weights = cls_weights / cls_weights.mean()
            
            # Assign weights using indices
            weights[source_indices] = cls_weights
        except Exception as e:
            print(f"Warning: Error computing weights for class {cls}: {e}")
            # Keep default weight of 1 for this class
    
    return weights
    


# without weights adaptive
# fit models
clf = LogisticRegression(random_state=0, class_weight='balanced', C=0.5)
clf.fit(X_train_scaled, y)

# make predictions
val_pred = clf.predict(X_val)
report = classification_report(y_val, val_pred)
print(f"="*50)
print(f"Resutls without weights adaptation")
print(f"Cross Entropy score: {weighted_log_loss_test(y_val, val_pred):.3f}")
print(report)

# adaptive weights model
# get class conditional weights
# For each class, compute domain weights separately
weights = weights = class_conditional_weights(
    source_X=X_train_scaled,  
    source_y=y.values.flatten(),
    target_X=X_val,  
    target_y=y_val.values.flatten(),
)

adpative_clf = LogisticRegression(random_state=0, class_weight='balanced', C=0.5)
adpative_clf.fit(X_train_scaled, y, sample_weight=weights)
adpative_pred = adpative_clf.predict(X_val)

print(f"="*50)
print(f"Resutls without weights adaptation")
print(f"Cross Entropy score: {weighted_log_loss_test(y_val, adpative_pred):.3f}")
print(classification_report(y_val, adpative_pred))




##
# Add new imports at the top
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def compare_class_performance(y_true, pred1, pred2):
    """Compare per-class F1 scores between two models"""
    # Get complete list of unique classes present in any report
    report1 = classification_report(y_true, pred1, output_dict=True, zero_division=0)
    report2 = classification_report(y_true, pred2, output_dict=True, zero_division=0)
    
    # Get all unique classes from both reports
    all_classes = set(report1.keys()).union(report2.keys())
    class_names = [str(k) for k in all_classes 
                  if k not in ('accuracy', 'macro avg', 'weighted avg')]
    
    # Sort classes numerically if possible
    try:
        class_names = sorted(class_names, key=lambda x: int(x))
    except ValueError:
        class_names = sorted(class_names)
    
    f1_diff = []
    for cls in class_names:
        # Handle missing classes in either report
        f1_1 = report1.get(cls, {}).get('f1-score', 0)
        f1_2 = report2.get(cls, {}).get('f1-score', 0)
        f1_diff.append(f1_2 - f1_1)
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(class_names, f1_diff)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.axhline(0, color='black', linestyle='--')
    plt.title('F1 Score Improvement by Class (Adaptive vs Baseline)')
    plt.ylabel('F1 Score Difference')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

compare_class_performance(y_val, val_pred, adpative_pred)


# ### Discussion about weights adaptation
# Based on the above result, the weight adaptation method improved performance of the model on the most of the classes that effectively migtigate the effect of the conditional shift on the 

# ### Feature distribution of example classes in training and test set



# Find features with largest distribution shifts
import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

def compute_kl_divergence(X_train_features, X_val_features):
    """
    Compute KL divergence between features of training and validation sets.
    
    Parameters:
    - X_train_features: Feature values from training set for a specific class
    - X_val_features: Feature values from validation set for the same class
    
    Returns:
    - Dictionary of KL divergence for each feature
    """
    n_features = X_train_features.shape[1]
    kl_divs = {}
    
    for feature_idx in range(n_features):
        # Extract feature values for training and validation
        train_values = X_train_features[:, feature_idx]
        val_values = X_val_features[:, feature_idx]
        
        # Skip if either dataset has too few samples
        if len(train_values) < 5 or len(val_values) < 5:
            kl_divs[feature_idx] = np.nan
            continue
            
        # Use KDE to estimate probability distributions
        # Define common grid for both distributions
        min_val = min(train_values.min(), val_values.min())
        max_val = max(train_values.max(), val_values.max())
        grid = np.linspace(min_val, max_val, 1000)
        
        # Fit KDE to both distributions
        kde_train = KernelDensity(bandwidth=0.1).fit(train_values.reshape(-1, 1))
        kde_val = KernelDensity(bandwidth=0.1).fit(val_values.reshape(-1, 1))
        
        # Get log density estimations
        pdf_train = np.exp(kde_train.score_samples(grid.reshape(-1, 1)))
        pdf_val = np.exp(kde_val.score_samples(grid.reshape(-1, 1)))
        
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        pdf_train = pdf_train + epsilon
        pdf_val = pdf_val + epsilon
        
        # Normalize to ensure they are valid probability distributions
        pdf_train = pdf_train / pdf_train.sum()
        pdf_val = pdf_val / pdf_val.sum()
        
        # Compute KL divergence
        kl = entropy(pdf_train, pdf_val)
        kl_divs[feature_idx] = kl
    
    return kl_divs

# Your original code block
critical_classes = np.arange(28)
all_feature_divs = {}

# For storing top features per class
top_features_per_class = {}

y_labels = y.values.flatten()
val_labels = y_val.values.flatten()
for cls in critical_classes:
    
    kl_divs = compute_kl_divergence(
        X_train_scaled[y_labels == cls], 
        X_val[val_labels == cls]
    )
    
    # Store results
    all_feature_divs[cls] = kl_divs
    
    # Find top 3 features with highest divergence
    sorted_features = sorted(kl_divs.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:3]
    top_features_per_class[cls] = top_features
    
    print(f"Class {cls} KL divergence summary:")
    for feature_idx, kl_value in top_features:
        if not np.isnan(kl_value):
            print(f"  Feature {feature_idx}: {kl_value:.4f}")

# Find features that consistently show high divergence across classes
feature_counts = {}
for cls, top_features in top_features_per_class.items():
    for feature_idx, _ in top_features:
        if feature_idx not in feature_counts:
            feature_counts[feature_idx] = 0
        feature_counts[feature_idx] += 1

# Print features that appear most frequently in the top 3
print("\nFeatures with highest distribution shifts across classes:")
sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
for feature_idx, count in sorted_features[:5]:
    print(f"Feature {feature_idx} appeared in top 3 for {count} classes")




def plot_class_conditional_subplots(feature_indices, class_labels, X_train, y_train, X_val, y_val, feature_names):
    """
    Create subplots comparing feature distributions for multiple classes and features.
    
    Args:
        feature_indices (list): List of two feature indices to plot
        class_labels (list): List of four class labels to compare
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array): Validation/test features
        y_val (array): Validation/test labels
        feature_names (list): List of feature names
    """
    plt.figure(figsize=(15, 20))
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # Flatten axes for easy indexing
    axes = axes.ravel()
    
    for idx, cls in enumerate(class_labels):
        # Get data masks
        train_mask = (y_train.values.flatten() == cls)
        val_mask = (y_val.values.flatten() == cls)
        
        # Plot first feature
        ax = axes[idx*2]
        if sum(train_mask) > 0:
            sns.kdeplot(X_train[train_mask, feature_indices[0]], 
                        ax=ax, label='Train', fill=True)
        if sum(val_mask) > 0:
            sns.kdeplot(X_val[val_mask, feature_indices[0]], 
                        ax=ax, label='Test Two Labelled', fill=True)
        ax.set_title(f'Class {cls}, Feature:{feature_names[feature_indices[0]]}')
        ax.legend()
        
        # Plot second feature
        ax = axes[idx*2 + 1]
        if sum(train_mask) > 0:
            sns.kdeplot(X_train[train_mask, feature_indices[1]], 
                        ax=ax, label='Train', fill=True)
        if sum(val_mask) > 0:
            sns.kdeplot(X_val[val_mask, feature_indices[1]], 
                        ax=ax, label='Test Two Labelled', fill=True)
        ax.set_title(f'Class {cls}, Feature:{feature_names[feature_indices[1]]}')
        ax.legend()
    plt.savefig(f'plots/example_class_distribution_comparison.png')
    plt.tight_layout()
    plt.show()

# Example usage
critical_classes = [9, 14,19, 21]  # 4 classes showing significant shift
example_features = [0, 2]        

plot_class_conditional_subplots(
    feature_indices=example_features,
    class_labels=critical_classes,
    X_train=X_train_scaled,
    y_train=y,
    X_val=X_val,
    y_val=y_val,
    feature_names=X_train_df.columns.tolist()
)




# Plot t-SNE before/after weighting
from sklearn.manifold import TSNE

def plot_tsne_comparison(X_train, y_train, X_test, y_test, weights=None, n_samples=1000):
    """
    Compare t-SNE visualizations of source and target distributions before/after weighting.
    
    Parameters:
    X_train, y_train: Source (training) data and labels
    X_test, y_test: Target (test) data and labels
    weights: Importance weights for source data (optional)
    n_samples: Number of points to sample for visualization
    """
    # Set up plot
    plt.figure(figsize=(18, 8))
    
    # Common parameters
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'random_state': 42
    }
    
    # Sample data consistently
    test_idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    if weights is not None:
        train_idx = np.random.choice(len(X_train), n_samples, p=weights/weights.sum())
    else:
        train_idx = np.random.choice(len(X_train), n_samples)
    
    # Create combined datasets
    combined_before = np.vstack([X_train[train_idx], X_test[test_idx]])
    combined_labels = np.hstack([y_train.iloc[train_idx].values, y_test.iloc[test_idx].values])
    domains = np.hstack([np.zeros(n_samples), np.ones(len(test_idx))])
    
    # Fit t-SNE once for both plots
    tsne = TSNE(**tsne_params)
    X_tsne = tsne.fit_transform(combined_before)
    
    # Before weighting
    plt.subplot(1, 2, 1)
    for domain in [0, 1]:
        mask = (domains == domain)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    c=combined_labels[mask], cmap='tab20',
                    marker='o' if domain == 0 else 's',
                    alpha=0.6, label='Train' if domain == 0 else 'Test')
    plt.title("Before Weighting\nColor=Class, Marker=Domain")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # After weighting (if weights provided)
    if weights is not None:
        plt.subplot(1, 2, 2)
        for domain in [0, 1]:
            mask = (domains == domain)
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                        c=combined_labels[mask], cmap='tab20',
                        marker='o' if domain == 0 else 's',
                        alpha=0.6, label='Train' if domain == 0 else 'Test')
        plt.title("After Class-Conditional Weighting\nColor=Class, Marker=Domain")
        plt.xlabel("t-SNE 1")
    
    # Add common elements
    plt.colorbar(plt.cm.ScalarMappable(cmap='tab20'), 
                 ax=plt.gcf().axes, label='Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Usage
plot_tsne_comparison(
    X_train_scaled, y,
    X_val, y_val,
    weights=weights
)




# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# def weighted_log_loss_test(y_true, y_pred):
#     """
#     Compute weighted log loss for multi-class classification.
    
#     Parameters:
#     - y_true: True labels
#     - y_pred: Predicted labels (not probabilities)
    
#     Returns:
#     - Weighted log loss value
#     """
#     # Convert inputs to numpy arrays if they aren't already
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
    
#     # Check if inputs are pandas Series/DataFrames
#     if hasattr(y_true, 'values'):
#         y_true = y_true.values
#     if hasattr(y_pred, 'values'):
#         y_pred = y_pred.values
    
#     # Get unique classes
#     classes = np.unique(y_true)
    
#     # Initialize loss
#     total_loss = 0
#     total_weight = 0
    
#     # Compute loss for each class
#     for cls in classes:
#         # Create binary indicators for this class
#         true_bin = (y_true == cls).astype(int)
#         pred_bin = (y_pred == cls).astype(int)
        
#         # Get class weight (proportion in validation set)
#         class_weight = np.sum(true_bin) / len(true_bin)
        
#         if class_weight > 0:  # Skip classes not present
#             # Calculate binary cross-entropy
#             epsilon = 1e-15  # To avoid log(0)
#             pred_bin = np.clip(pred_bin, epsilon, 1 - epsilon)
#             loss = -(true_bin * np.log(pred_bin) + (1 - true_bin) * np.log(1 - pred_bin))
            
#             # Weight by class proportion
#             weighted_loss = class_weight * np.mean(loss)
#             total_loss += weighted_loss
#             total_weight += class_weight
    
#     # Normalize by total weight
#     if total_weight > 0:
#         final_loss = total_loss / total_weight
#     else:
#         final_loss = 0
        
#     return final_loss

# For GridSearchCV, we need a scoring function that handles the different way scikit-learn passes predictions
def weighted_log_loss_scorer(estimator, X, y):
    """
    Custom scorer for GridSearchCV using weighted_log_loss_test.
    This function gets the actual estimator, not just predictions.
    """
    # Get predictions (not probabilities)
    y_pred = estimator.predict(X)
    
    # Calculate loss (negative because GridSearchCV maximizes)
    return -weighted_log_loss_test(y, y_pred)

# Define parameter grids for different penalty types
l2_params = {
    'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'max_iter': [2000]
}

l1_params = {
    'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'max_iter': [2000]
}

elasticnet_params = {
    'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],
    'max_iter': [2000]
}

# Train models with different penalties
print("Training L2 penalty models...")
l2_grid = GridSearchCV(
    LogisticRegression(random_state=0, class_weight='balanced', penalty='l2', solver='lbfgs'),
    l2_params,
    scoring=weighted_log_loss_scorer,  # Use the function directly, not make_scorer
    cv=5,
    n_jobs=-1
)
l2_grid.fit(X_train_scaled, y, sample_weight=weights)

print("Training L1 penalty models...")
l1_grid = GridSearchCV(
    LogisticRegression(random_state=0, class_weight='balanced', penalty='l1', solver='saga'),
    l1_params,
    scoring=weighted_log_loss_scorer,  # Use the function directly, not make_scorer
    cv=5,
    n_jobs=-1
)
l1_grid.fit(X_train_scaled, y, sample_weight=weights)

print("Training Elasticnet penalty models...")
elasticnet_grid = GridSearchCV(
    LogisticRegression(random_state=0, class_weight='balanced', penalty='elasticnet', solver='saga'),
    elasticnet_params,
    scoring=weighted_log_loss_scorer,  # Use the function directly, not make_scorer
    cv=5,
    n_jobs=-1
)
elasticnet_grid.fit(X_train_scaled, y, sample_weight=weights)




# Find best model across all penalties
models = [
    ("L2", l2_grid),
    ("L1", l1_grid),
    ("Elasticnet", elasticnet_grid)
]

best_score = float('-inf')
best_model = None
best_params = None
best_name = None

for name, grid in models:
    score = -grid.best_score_  # Convert back to positive
    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_name = name

print(f"\nBest overall model: {best_name}")
print(f"Best parameters: {best_params}")
print(f"Best CV score: {best_score:.4f}")

# Evaluate best model
adaptive_pred = best_model.predict(X_val)
print(f"='*50")
print(f"Results with optimized model")
print(f"Cross Entropy score: {weighted_log_loss_test(y_val, adaptive_pred):.4f}")
print(classification_report(y_val, adaptive_pred))

# Compare with original model
print(f"='*50")
print(f"Original model results")
original_clf = LogisticRegression(random_state=0, class_weight='balanced', C=0.5)
original_clf.fit(X_train_scaled, y, sample_weight=weights)
original_pred = original_clf.predict(X_val)
print(f"Cross Entropy score: {weighted_log_loss_test(y_val, original_pred):.4f}")
print(classification_report(y_val, original_pred))




# Compare with original model
print(f"="*50)
print(f"Fine tuned model results")
trial_clf = LogisticRegression(random_state=0, class_weight='balanced', C=11, penalty='l2', solver='sag')
trial_clf.fit(X_train_scaled, y, sample_weight=weights)
trial_pred = trial_clf.predict(X_val)
print(f"Cross Entropy score: {weighted_log_loss_test(y_val, trial_pred):.4f}")
print(classification_report(y_val, trial_pred))




# random_state=0, class_weight='balanced', C=28



# trial_clf = LogisticRegression(random_state=0, class_weight='balanced', C=32, penalty='l2')
# Cross Entropy score: 0.0174
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.44      1.00      0.62         4
#            5       1.00      1.00      1.00         9
#            6       0.60      1.00      0.75         3
#            7       0.83      0.71      0.77        14
#            8       0.88      0.79      0.84        29
#            9       1.00      0.33      0.50         3
#           10       0.67      1.00      0.80         2
#           11       1.00      1.00      1.00         9
#           12       0.86      0.72      0.78        43
#           13       0.14      1.00      0.25         1
#           14       1.00      1.00      1.00         6
#           15       1.00      1.00      1.00         1
#           17       0.90      0.90      0.90        10
#           18       0.00      0.00      0.00         1
#           19       0.60      0.75      0.67         4
#           20       0.67      0.67      0.67         3
#           21       1.00      1.00      1.00         7
#           23       0.33      1.00      0.50         1
#           24       0.59      0.67      0.62        15
#           25       0.89      0.65      0.76        26
#           26       0.20      1.00      0.33         1
#           27       1.00      0.88      0.93         8

#     accuracy                           0.78       202
#    macro avg       0.65      0.75      0.65       202
# weighted avg       0.83      0.78      0.79       202



# trial_clf = LogisticRegression(random_state=0, class_weight='balanced', penalty='elasticnet',l1_ratio=0.1, C=28, solver ='saga')
# Cross Entropy score: 0.0187
#               precision    recall  f1-score   support

#            2       0.25      1.00      0.40         1
#            3       0.00      0.00      0.00         1
#            4       0.40      1.00      0.57         4
#            5       1.00      1.00      1.00         9
#            6       0.60      1.00      0.75         3
#            7       0.85      0.79      0.81        14
#            8       0.86      0.83      0.84        29
#            9       0.00      0.00      0.00         3
#           10       0.67      1.00      0.80         2
#           11       1.00      0.89      0.94         9
#           12       0.91      0.70      0.79        43
#           13       0.33      1.00      0.50         1
#           14       0.86      1.00      0.92         6
#           15       1.00      1.00      1.00         1
#           17       0.91      1.00      0.95        10
#           18       0.00      0.00      0.00         1
#           19       0.50      0.50      0.50         4
#           20       0.50      0.67      0.57         3
#           21       1.00      1.00      1.00         7
#           23       0.25      1.00      0.40         1
#           24       0.56      0.67      0.61        15
#           25       0.95      0.69      0.80        26
#           26       0.00      0.00      0.00         1
#           27       1.00      0.88      0.93         8

#     accuracy                           0.78       202
#    macro avg       0.60      0.73      0.63       202
# weighted avg       0.82      0.78      0.79       202


# trial_clf = LogisticRegression(random_state=0, class_weight='balanced', penalty='elasticnet',l1_ratio=0.1, C=30, solver ='saga')
# Cross Entropy score: 0.0222
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.50      1.00      0.67         4
#            5       1.00      0.89      0.94         9
#            6       0.43      1.00      0.60         3
#            7       0.83      0.71      0.77        14
#            8       0.89      0.83      0.86        29
#            9       0.00      0.00      0.00         3
#           10       0.67      1.00      0.80         2
#           11       1.00      1.00      1.00         9
#           12       0.82      0.77      0.80        43
#           13       0.09      1.00      0.17         1
#           14       1.00      1.00      1.00         6
#           15       1.00      1.00      1.00         1
#           17       1.00      0.90      0.95        10
#           18       0.00      0.00      0.00         1
#           19       0.50      0.50      0.50         4
#           20       0.29      0.67      0.40         3
#           21       1.00      1.00      1.00         7
#           23       0.20      1.00      0.33         1
#           24       0.71      0.67      0.69        15
#           25       1.00      0.50      0.67        26
#           26       0.00      0.00      0.00         1
#           27       1.00      0.88      0.93         8

#     accuracy                           0.75       202
#    macro avg       0.58      0.68      0.59       202
# weighted avg       0.83      0.75      0.77       202






import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel

# Assuming X_train_scaled and X_val_scaled are properly scaled versions
# First identify redundant features using correlation analysis

# Convert scaled data to DataFrame
X_train_df = pd.DataFrame(X_train_scaled)

# Calculate correlation matrix
corr_matrix = X_train_df.corr().abs()

# Select upper triangle of correlation matrix
upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_tri = corr_matrix.where(upper)

# Find features with correlation above threshold (0.9)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

# Get column indices to drop
to_drop_indices = [int(col) for col in to_drop]

# Remove redundant features from both training and validation sets
X_train_reduced = np.delete(X_train_scaled, to_drop_indices, axis=1)


# Now train and evaluate with reduced features
print(f"="*50)
print(f"Reduced feature model results")
trial_clf = LogisticRegression(random_state=0, class_weight='balanced', C=28, penalty='l2')
trial_clf.fit(X_train_reduced, y, sample_weight=weights)
trial_pred = trial_clf.predict(X_val_reduced)

print(f"Cross Entropy score: {weighted_log_loss_test(y_val, trial_pred):.4f}")
print(classification_report(y_val, trial_pred))




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report


# 3. Initial model training with all features
base_clf = LogisticRegression(
    random_state=0,
    class_weight='balanced',
    penalty='elasticnet',
    l1_ratio=0.1,
    C=30,
    solver='saga',
    max_iter=10000
)
base_clf.fit(X_train_scaled, y, sample_weight=weights)

# 4. Model-based feature selection
# Select features with coefficients above median absolute value
selector = SelectFromModel(
    estimator=base_clf,
    prefit=True,
    threshold='median'
)

X_train_reduced = selector.transform(X_train_scaled)
X_val_reduced = selector.transform(X_val)

# 5. Train final model with selected features
final_clf = LogisticRegression(
    random_state=0,
    class_weight='balanced',
    penalty='elasticnet',
    l1_ratio=0.1,
    C=30,
    solver='saga',
    max_iter=10000
)

final_clf.fit(X_train_reduced, y, sample_weight=weights)

# 6. Evaluate both models
def evaluate_model(model, X_train, X_val, y_train, y_val):
    print(f"Number of features: {X_train.shape[1]}")
    print("Training scores:")
    train_pred = model.predict(X_train)
    print(classification_report(y_train, train_pred))
    
    print("Validation scores:")
    val_pred = model.predict(X_val)
    print(classification_report(y_val, val_pred))
    print("="*50)

print("Original Model Evaluation:")
evaluate_model(base_clf, X_train_scaled, X_val, y, y_val)

print("Reduced Feature Model Evaluation:")
evaluate_model(final_clf, X_train_reduced, X_val_reduced, y, y_val)

# Optional: Get selected feature names if using DataFrame
selected_features = selector.get_support()
if isinstance(X, pd.DataFrame):
    feature_names = X.columns[selected_features]
    print(f"Selected features ({len(feature_names)}): {', '.join(feature_names)}")







if __name__ == "__main__":
    # TODO: 调用你想执行的主函数，比如 main()
    pass
