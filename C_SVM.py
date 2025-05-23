# -*- coding: utf-8 -*-
"""C-SVM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B9JeUnHTeRtdbsoRLhbYmj_2oWzAnuj6
"""

# Weighted CE score for evaluate the methods results
def weighted_log_loss(y_true_np, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.

    Parameters:
    - y_true: 1-d array like object contains (range 0-27), shape: (n_samples,)
    - y_pred: array like object contains list of probabilities, shape: (n_samples, 28)

    Returns:
    - Weighted log loss (scalar).
    """
    import numpy as np
    import pandas as pd

    # Number of classes
    n_classes = 28

    # Convert discrete labels to one-hot encoded format
    def to_one_hot(labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            if 0 <= label < num_classes:  # Safety check
                one_hot[i, label] = 1
        return one_hot

    # Convert true labels to one-hot format
    y_true_one_hot = to_one_hot(y_true_np, n_classes)


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

    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

    # Calculate log loss term
    log_terms = np.sum(y_true_one_hot * np.log(y_pred), axis=1)
    loss = -np.mean(sample_weights * log_terms)

    return loss

def evaluate_model(model, X, y_true):
    """
        model: fitted model
        X: feature matrix (n_sample, 300)
        y_true: array, contains true labels (0-27), (n_samples, )
    """
    y_pred_prob = model.predict_proba(X)  # (n_samples, 28)
    y_pred_label = model.predict(X)
    ce = weighted_log_loss(y_true, y_pred_prob)
    print(f"Weighted CE: {ce:.4f}")
    print(classification_report(y_true, y_pred_label, zero_division=0))
    return ce

# 5-fold cv on train set

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

def weighted_log_loss(y_true_np, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.

    Parameters:
    - y_true: 1-d array like object contains (range 0-27), shape: (n_samples,)
    - y_pred: array like object contains list of probabilities, shape: (n_samples, 28)

    Returns:
    - Weighted log loss (scalar).
    """
    import numpy as np
    import pandas as pd

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
    log_terms = np.sum(y_true_one_hot * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)), axis=1)
    loss = -np.mean(sample_weights * log_terms)

    return loss

def create_full_probability_matrix(local_probabilities, local_classes, num_total_classes=28):
    """
    Creates a full probability matrix (size n_samples x num_total_classes)
    from the local probabilities predicted by a cluster-specific SVM.

    Parameters:
    - local_probabilities: Array of probabilities predicted by the cluster SVM.
                           Shape: (n_samples, n_local_classes)
    - local_classes: List of the original class labels that were present in the cluster's training data.
    - num_total_classes: The total number of classes in the original problem (default=28).

    Returns:
    - full_probs: Array of full probabilities. Shape: (n_samples, num_total_classes)
    """
    n_samples = local_probabilities.shape[0]
    full_probs = np.zeros((n_samples, num_total_classes))
    for i, local_prob in enumerate(local_probabilities):
        for j, local_class_index in enumerate(range(len(local_classes))):
            original_class = local_classes[local_class_index]
            if 0 <= original_class < num_total_classes:
                full_probs[i, original_class] = local_prob[j]
    return full_probs

# Load data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- Hierarchical Clustering of Classes (Done Once) ---
class_means = []; unique_classes = np.unique(y_train)
for cls in unique_classes: class_means.append(X_train_scaled[y_train == cls].mean(axis=0))
class_means = np.array(class_means)
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
class_clusters = agg_clustering.fit_predict(class_means)
class_to_cluster = {cls: cluster for cls, cluster in zip(unique_classes, class_clusters)}

# --- Prepare Cluster Labels ---
cluster_labels = np.array([class_to_cluster[label] for label in y_train])

# --- Best Parameters for Individual SVMs ---
best_parameters = {
    0: {'C': 5, 'gamma': 0.25},
    1: {'C': 15, 'gamma': 0.5},
    2: {'C': 15, 'gamma': 0.25},
    3: {'C': 5, 'gamma': 0.25}
}

# --- 5-Fold Stratified Cross-Validation ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_true_labels = []
all_predictions = []
all_prob_matrices = []

for fold, (train_index, test_index) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"\n--- Fold {fold + 1} ---")

    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    cluster_labels_train, cluster_labels_test = cluster_labels[train_index], cluster_labels[test_index]

    # --- Train the 4-Cluster Classifier SVM ---
    cluster_assignment_svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
    cluster_assignment_svm.fit(X_train_fold, cluster_labels_train)

    # --- Train the Four Individual Cluster-Specific SVMs ---
    cluster_models = {}
    cluster_local_classes = {}
    for cluster_id in range(4):
        cluster_train_mask = np.isin(y_train_fold, [cls for cls in unique_classes if class_to_cluster[cls] == cluster_id])
        X_cluster_train_fold = X_train_fold[cluster_train_mask]
        y_cluster_train_fold = y_train_fold[cluster_train_mask]
        local_classes = np.unique(y_cluster_train_fold)
        cluster_local_classes[cluster_id] = local_classes

        if len(X_cluster_train_fold) > 0:
            svm = SVC(kernel='rbf', C=best_parameters[cluster_id]['C'],
                      gamma=best_parameters[cluster_id]['gamma'], class_weight='balanced',
                      random_state=42, probability=True)
            svm.fit(X_cluster_train_fold, y_cluster_train_fold)
            cluster_models[cluster_id] = svm
        else:
            print(f"Warning: No training data for Cluster {cluster_id} in this fold.")
            cluster_models[cluster_id] = None

    # --- Evaluate on the Test Set of the Current Fold ---
    fold_predictions = []
    fold_true_labels = list(y_test_fold)
    fold_prob_matrices = []

    for i in range(len(X_test_fold)):
        unseen_instance = X_test_fold[i].reshape(1, -1)
        true_label = y_test_fold[i]

        # Predict the cluster for the unseen instance
        predicted_cluster = cluster_assignment_svm.predict(unseen_instance)[0]

        # Get the corresponding cluster-specific SVM and predict the final class
        if predicted_cluster in cluster_models and cluster_models[predicted_cluster] is not None and hasattr(cluster_models[predicted_cluster], 'predict_proba'):
            local_probabilities = cluster_models[predicted_cluster].predict_proba(unseen_instance)
            local_classes = cluster_local_classes.get(predicted_cluster, [])
            full_prob_matrix = create_full_probability_matrix(local_probabilities, local_classes)

            final_prediction = np.argmax(full_prob_matrix)

            fold_predictions.append(final_prediction)
            fold_prob_matrices.append(full_prob_matrix[0])
        else:
            print(f"Warning: No model or no probability prediction for predicted cluster {predicted_cluster} for instance with true label {true_label} in this fold.")
            fold_predictions.append(-1)
            # Placeholder
            fold_prob_matrices.append(np.zeros(28))

    all_true_labels.extend(fold_true_labels)
    all_predictions.extend(fold_predictions)
    all_prob_matrices.extend(fold_prob_matrices)

# --- Final Evaluation ---
print("\n--- Overall Classification Report (5-Fold CV) ---")
print(classification_report(all_true_labels, all_predictions, zero_division=0))

# Calculate overall weighted log loss
overall_weighted_loss = weighted_log_loss(np.array(all_true_labels), np.array(all_prob_matrices))
print(f"\nOverall Weighted Log Loss (5-Fold CV): {overall_weighted_loss:.4f}")

# evaluate on test_2

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

def weighted_log_loss(y_true_np, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.

    Parameters:
    - y_true: 1-d array like object contains (range 0-27), shape: (n_samples,)
    - y_pred: array like object contains list of probabilities, shape: (n_samples, 28)

    Returns:
    - Weighted log loss (scalar).
    """
    import numpy as np
    import pandas as pd

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
    log_terms = np.sum(y_true_one_hot * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)), axis=1)
    loss = -np.mean(sample_weights * log_terms)

    return loss

def create_full_probability_matrix(local_probabilities, local_classes, num_total_classes=28):
    """
    Creates a full probability matrix (size n_samples x num_total_classes)
    from the local probabilities predicted by a cluster-specific SVM.

    Parameters:
    - local_probabilities: Array of probabilities predicted by the cluster SVM.
                          Shape: (n_samples, n_local_classes)
    - local_classes: List of the original class labels that were present in the cluster's training data.
    - num_total_classes: The total number of classes in the original problem (default=28).

    Returns:
    - full_probs: Array of full probabilities. Shape: (n_samples, num_total_classes)
    """
    n_samples = local_probabilities.shape[0]
    full_probs = np.zeros((n_samples, num_total_classes))
    for i, local_prob in enumerate(local_probabilities):
        for j, local_class_index in enumerate(range(len(local_classes))):
            original_class = local_classes[local_class_index]
            if 0 <= original_class < num_total_classes:
                full_probs[i, original_class] = local_prob[j]
    return full_probs

# Load training data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Scale training features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- Hierarchical Clustering of Classes (Done Once) ---
class_means = []; unique_classes = np.unique(y_train)
for cls in unique_classes: class_means.append(X_train_scaled[y_train == cls].mean(axis=0))
class_means = np.array(class_means)
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
class_clusters = agg_clustering.fit_predict(class_means)
class_to_cluster = {cls: cluster for cls, cluster in zip(unique_classes, class_clusters)}

# --- Prepare Cluster Labels for Entire Training Set ---
cluster_labels_train_full = np.array([class_to_cluster[label] for label in y_train])

# --- Best Parameters for Individual SVMs ---
best_parameters = {
    0: {'C': 5, 'gamma': 0.25},
    1: {'C': 15, 'gamma': 0.5},
    2: {'C': 15, 'gamma': 0.25},
    3: {'C': 5, 'gamma': 0.25}
}

# --- Train the 4-Cluster Classifier SVM on the Entire Training Set ---
cluster_assignment_svm_full = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
cluster_assignment_svm_full.fit(X_train_scaled, cluster_labels_train_full)

# --- Train the Four Individual Cluster-Specific SVMs on the Entire Training Set ---
cluster_models_full = {}
cluster_local_classes_full = {}
for cluster_id in range(4):
    cluster_train_mask_full = np.isin(y_train, [cls for cls in unique_classes if class_to_cluster[cls] == cluster_id])
    X_cluster_train_full = X_train_scaled[cluster_train_mask_full]
    y_cluster_train_full = y_train[cluster_train_mask_full]
    local_classes_full = np.unique(y_cluster_train_full)
    cluster_local_classes_full[cluster_id] = local_classes_full

    if len(X_cluster_train_full) > 0:
        svm_full = SVC(kernel='rbf', C=best_parameters[cluster_id]['C'],
                        gamma=best_parameters[cluster_id]['gamma'], class_weight='balanced',
                        random_state=42, probability=True)
        svm_full.fit(X_cluster_train_full, y_cluster_train_full)
        cluster_models_full[cluster_id] = svm_full
    else:
        print(f"Warning: No training data for Cluster {cluster_id} in the full training set.")
        cluster_models_full[cluster_id] = None

# --- Load and Prepare Test Data ---
X_test_2 = pd.read_csv('X_test_2.csv').head(202)
y_test_2_reduced = pd.read_csv('y_test_2_reduced.csv').values.ravel()

# Scale test features using the scaler fitted on the training data
X_test_2_scaled = scaler.transform(X_test_2)

# --- Evaluate on the Loaded Test Data ---
test_predictions = []
test_prob_matrices = []

for i in range(len(X_test_2_scaled)):
    unseen_instance = X_test_2_scaled[i].reshape(1, -1)

    # Predict the cluster for the unseen instance
    predicted_cluster = cluster_assignment_svm_full.predict(unseen_instance)[0]

    # Get the corresponding cluster-specific SVM and predict the final class
    if predicted_cluster in cluster_models_full and cluster_models_full[predicted_cluster] is not None and hasattr(cluster_models_full[predicted_cluster], 'predict_proba'):
        local_probabilities = cluster_models_full[predicted_cluster].predict_proba(unseen_instance)
        local_classes = cluster_local_classes_full.get(predicted_cluster, [])
        full_prob_matrix = create_full_probability_matrix(local_probabilities, local_classes)
        final_prediction = np.argmax(full_prob_matrix)
        test_predictions.append(final_prediction)
        test_prob_matrices.append(full_prob_matrix[0])
    else:
        print(f"Warning: No model or no probability prediction for predicted cluster {predicted_cluster} for test instance {i}.")
        test_predictions.append(-1)
        test_prob_matrices.append(np.zeros(28)) # Placeholder

# --- Final Evaluation on Test Data ---
print("\n--- Evaluation on X_test_2 (first 202 instances) and y_test_2_reduced ---")
print(classification_report(y_test_2_reduced, test_predictions, zero_division=0))

# Calculate weighted log loss on the test data
overall_weighted_loss_test = weighted_log_loss(y_test_2_reduced, np.array(test_prob_matrices))
print(f"\nWeighted Log Loss on Test Data: {overall_weighted_loss_test:.4f}")

# 6-fold cv on combined data sets

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

def weighted_log_loss(y_true_np, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.

    Parameters:
    - y_true: 1-d array like object contains (range 0-27), shape: (n_samples,)
    - y_pred: array like object contains list of probabilities, shape: (n_samples, 28)

    Returns:
    - Weighted log loss (scalar).
    """
    import numpy as np
    import pandas as pd

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
    log_terms = np.sum(y_true_one_hot * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)), axis=1)
    loss = -np.mean(sample_weights * log_terms)

    return loss

def create_full_probability_matrix(local_probabilities, local_classes, num_total_classes=28):
    """
    Creates a full probability matrix (size n_samples x num_total_classes)
    from the local probabilities predicted by a cluster-specific SVM.

    Parameters:
    - local_probabilities: Array of probabilities predicted by the cluster SVM.
                           Shape: (n_samples, n_local_classes)
    - local_classes: List of the original class labels that were present in the cluster's training data.
    - num_total_classes: The total number of classes in the original problem (default=28).

    Returns:
    - full_probs: Array of full probabilities. Shape: (n_samples, num_total_classes)
    """
    n_samples = local_probabilities.shape[0]
    full_probs = np.zeros((n_samples, num_total_classes))
    for i, local_prob in enumerate(local_probabilities):
        for j, local_class_index in enumerate(range(len(local_classes))):
            original_class = local_classes[local_class_index]
            if 0 <= original_class < num_total_classes:
                full_probs[i, original_class] = local_prob[j]
    return full_probs

# Load data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Load the first 202 rows of the new test data
X_test_2_reduced = pd.read_csv('X_test_2.csv', nrows=202)
y_test_2_reduced = pd.read_csv('y_test_2_reduced.csv').values.ravel()

# Append the new test data to the training data
X_train = pd.concat([X_train, X_test_2_reduced], ignore_index=True)
y_train = pd.concat([pd.Series(y_train), pd.Series(y_test_2_reduced)], ignore_index=True).values

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# --- Hierarchical Clustering of Classes (Done Once) ---
class_means = []; unique_classes = np.unique(y_train)
for cls in unique_classes: class_means.append(X_train_scaled[y_train == cls].mean(axis=0))
class_means = np.array(class_means)
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
class_clusters = agg_clustering.fit_predict(class_means)
class_to_cluster = {cls: cluster for cls, cluster in zip(unique_classes, class_clusters)}

# --- Prepare Cluster Labels ---
cluster_labels = np.array([class_to_cluster[label] for label in y_train])

# --- Best Parameters for Individual SVMs ---
best_parameters = {
    0: {'C': 5, 'gamma': 0.25},
    1: {'C': 15, 'gamma': 0.5},
    2: {'C': 15, 'gamma': 0.25},
    3: {'C': 5, 'gamma': 0.25}
}

# --- 5-Fold Stratified Cross-Validation ---
n_splits = 6
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

all_true_labels = []
all_predictions = []
all_prob_matrices = []

for fold, (train_index, test_index) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"\n--- Fold {fold + 1} ---")

    X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    cluster_labels_train, cluster_labels_test = cluster_labels[train_index], cluster_labels[test_index]

    # --- Train the 4-Cluster Classifier SVM ---
    cluster_assignment_svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
    cluster_assignment_svm.fit(X_train_fold, cluster_labels_train)

    # --- Train the Four Individual Cluster-Specific SVMs ---
    cluster_models = {}
    cluster_local_classes = {}
    for cluster_id in range(4):
        cluster_train_mask = np.isin(y_train_fold, [cls for cls in unique_classes if class_to_cluster[cls] == cluster_id])
        X_cluster_train_fold = X_train_fold[cluster_train_mask]
        y_cluster_train_fold = y_train_fold[cluster_train_mask]
        local_classes = np.unique(y_cluster_train_fold)
        cluster_local_classes[cluster_id] = local_classes

        if len(X_cluster_train_fold) > 0:
            svm = SVC(kernel='rbf', C=best_parameters[cluster_id]['C'],
                      gamma=best_parameters[cluster_id]['gamma'], class_weight='balanced',
                      random_state=42, probability=True)
            svm.fit(X_cluster_train_fold, y_cluster_train_fold)
            cluster_models[cluster_id] = svm
        else:
            print(f"Warning: No training data for Cluster {cluster_id} in this fold.")
            cluster_models[cluster_id] = None

    # --- Evaluate on the Test Set of the Current Fold ---
    fold_predictions = []
    fold_true_labels = list(y_test_fold)
    fold_prob_matrices = []

    for i in range(len(X_test_fold)):
        unseen_instance = X_test_fold[i].reshape(1, -1)
        true_label = y_test_fold[i]

        # Predict the cluster for the unseen instance
        predicted_cluster = cluster_assignment_svm.predict(unseen_instance)[0]

        # Get the corresponding cluster-specific SVM and predict the final class
        if predicted_cluster in cluster_models and cluster_models[predicted_cluster] is not None and hasattr(cluster_models[predicted_cluster], 'predict_proba'):
            local_probabilities = cluster_models[predicted_cluster].predict_proba(unseen_instance)
            local_classes = cluster_local_classes.get(predicted_cluster, [])
            full_prob_matrix = create_full_probability_matrix(local_probabilities, local_classes)

            final_prediction = np.argmax(full_prob_matrix)

            fold_predictions.append(final_prediction)
            fold_prob_matrices.append(full_prob_matrix[0])
        else:
            print(f"Warning: No model or no probability prediction for predicted cluster {predicted_cluster} for instance with true label {true_label} in this fold.")
            fold_predictions.append(-1)
            fold_prob_matrices.append(np.zeros(28)) # Placeholder

    all_true_labels.extend(fold_true_labels)
    all_predictions.extend(fold_predictions)
    all_prob_matrices.extend(fold_prob_matrices)

# --- Final Evaluation ---
print("\n--- Overall Classification Report (5-Fold CV) ---")
print(classification_report(all_true_labels, all_predictions, zero_division=0))

# Calculate overall weighted log loss
overall_weighted_loss = weighted_log_loss(np.array(all_true_labels), np.array(all_prob_matrices))
print(f"\nOverall Weighted Log Loss (5-Fold CV): {overall_weighted_loss:.4f}")


#
# Investigating effect of kernelPCA
#

def weighted_log_loss(y_true_np, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.

    Parameters:
    - y_true: 1-d array like object contains (range 0-27), shape: (n_samples,)
    - y_pred: array like object contains list of probabilities, shape: (n_samples, 28)

    Returns:
    - Weighted log loss (scalar).
    """
    import numpy as np
    import pandas as pd

    # Number of classes
    n_classes = 28  # Classes 0-27

    # Convert discrete labels to one-hot encoded format
    def to_one_hot(labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            if 0 <= label < num_classes:
                one_hot[i, label] = 1
        return one_hot

    # Convert true labels to one-hot format
    y_true_one_hot = to_one_hot(y_true_np, n_classes)


    # Compute class frequencies
    class_counts = np.sum(y_true_one_hot, axis=0)

    # Compute class weights with safety check for zero counts
    class_weights = np.zeros_like(class_counts)
    for c in range(n_classes):
        # Avoid division by zero
        if class_counts[c] > 0:
            class_weights[c] = 1.0 / class_counts[c]

    # Normalize weights to sum to 1
    class_weights /= np.sum(class_weights)

    # Compute weighted loss
    sample_weights = np.sum(y_true_one_hot * class_weights, axis=1)

    # Calculate log loss term
    log_terms = np.sum(y_true_one_hot * np.log(y_pred), axis=1)
    loss = -np.mean(sample_weights * log_terms)

    return loss

# SVM, but without kernel PCA, just rbf

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load the data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Assuming 'label' column in y_train
y_train = y_train['label'].values

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
weighted_log_losses = []

for train_index, val_index in kf.split(X_train_scaled):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # SVM with RBF kernel
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_fold, y_train_fold)

    # Predict probabilities
    y_pred_prob = svm.predict_proba(X_val_fold)

    # Calculate weighted log loss
    loss = weighted_log_loss(y_val_fold, y_pred_prob)
    weighted_log_losses.append(loss)

# Print results
print(f"Weighted Log Loss for each fold: {weighted_log_losses}")
print(f"Average Weighted Log Loss: {np.mean(weighted_log_losses)}")

# 5 fold CV, SVM with RBF kernelpca

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# Load the data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# Assuming 'label' column in y_train
y_train = y_train['label'].values

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Kernel PCA
kpca = KernelPCA(n_components=50, kernel='rbf')
X_train_kpca = kpca.fit_transform(X_train_scaled)

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
weighted_log_losses = []

for train_index, val_index in kf.split(X_train_kpca):
    X_train_fold, X_val_fold = X_train_kpca[train_index], X_train_kpca[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # SVM with RBF kernel
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_fold, y_train_fold)

    # Predict probabilities
    y_pred_prob = svm.predict_proba(X_val_fold)

    # Calculate weighted log loss
    loss = weighted_log_loss(y_val_fold, y_pred_prob)
    weighted_log_losses.append(loss)

# Print results
print(f"Weighted Log Loss for each fold: {weighted_log_losses}")
print(f"Average Weighted Log Loss: {np.mean(weighted_log_losses)}")
