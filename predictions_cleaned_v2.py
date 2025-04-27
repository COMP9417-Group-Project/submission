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
y_train = pd.read_csv(y_train_path)
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


# define scaler
scaler = MinMaxScaler()
scaler.fit(X_train_df)

# transform data
X_train_scaled = scaler.transform(X_train_df)
X_test_two_scaled_labelled = scaler.transform(X_test_two_labelled)
X_test_two_scaled_unlabelled = scaler.transform(X_test_two_unlabelled)
X_test_one_scaled = scaler.transform(X_test_one)

# weighted log loss score calculation method
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
    print(f"Weighted CE: {ce:.5f}")
    print(classification_report(y_true, y_pred_label))
    return ce


# ### Weight Adaptations



def class_conditional_weights(source_X, source_y, target_X, target_y):
    """
    Calculate class-conditional importance weights for domain adaptation.
    """
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

# adapt weights
weights = class_conditional_weights(
    source_X=X_train_scaled,  
    source_y=y_train.values.flatten(),
    target_X=X_test_two_scaled_labelled,  
    target_y=y_test_two_label.values.flatten(),
)


# ### Adapted Class-Specific Ensembles(ACSE) For Test 2



import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier

class ClassSpecificEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, sgd_model, lr_model, sgd_classes):
        self.sgd_model = sgd_model  # Must be fitted before ensemble.fit()
        self.lr_model = lr_model    # Must be fitted before ensemble.fit()
        self.sgd_classes = sgd_classes
    
    def fit(self, X, y):
        # Verify models are already fitted and classes match
        if not hasattr(self.lr_model, 'classes_'):
            raise ValueError("LogisticRegression model must be fitted first.")
        if not hasattr(self.sgd_model, 'classes_'):
            raise ValueError("SGD model must be fitted first.")
        if not np.array_equal(self.lr_model.classes_, self.sgd_model.classes_):
            raise ValueError("SGD and LogisticRegression have different class labels.")
        self.classes_ = self.lr_model.classes_  # Set after validation
        return self
    
    def predict_proba(self, X):
        sgd_proba = self.sgd_model.predict_proba(X)
        lr_proba = self.lr_model.predict_proba(X)
        combined_proba = lr_proba.copy()
        combined_proba[:, self.sgd_classes] = sgd_proba[:, self.sgd_classes]
        combined_proba /= combined_proba.sum(axis=1, keepdims=True)  # Renormalize
        return combined_proba
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# Define classes to prioritize with SGD
sgd_target_classes = [4, 13, 26] 

# Train models 
best_lg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, C=0.7, penalty='l2')
best_lg.fit(X_train_scaled, y_train, sample_weight=weights)
best_sgd = SGDClassifier(random_state=0, loss='log_loss', tol=1e-3 )
best_sgd.fit(X_train_scaled, y_train, sample_weight=weights)


# Create ensemble
ensemble = ClassSpecificEnsemble(
    sgd_model=best_sgd,
    lr_model=best_lg,
    sgd_classes=sgd_target_classes
)
ensemble.fit(X_train_scaled, y_train) 
ensemble_ce = evaluate_model(ensemble, X_test_two_scaled_labelled, y_test_two_label.values.flatten())




# make predictions for submission
preds_test2 = ensemble.predict_proba(X_test_two_scaled_unlabelled) 
print(preds_test2.shape)

# Save predictions to a .npy file
np.save('preds_1.npy', preds_test2)


# ### Semi-supervised data integration for Test 1



# ============================== Model Evaluation Function =====================
def train_and_evaluate_model(X_train, y_train, X_val, y_val, model):
    model.fit(X_train, y_train)
    return evaluate_model(model, X_val, y_val)

# create pseudo labels from the best ensembles 
pseudo_set2_labels = ensemble.predict(X_test_two_scaled_unlabelled)

# integrate training set 
X_comb_pseudo = np.concatenate([X_train_scaled, X_test_two_scaled_unlabelled], axis=0)
y_comb_pseudo = np.concatenate([y_train.values.flatten(), pseudo_set2_labels], axis=0)

# define best model, and fit combined data
best_test_model= LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, C=0.7, penalty='l2' )
best_test_model.fit(X_comb_pseudo,y_comb_pseudo)

# make predictions for submission
preds_test1 = best_test_model.predict_proba(X_test_one_scaled)
print(preds_test1.shape)

# Save predictions to a .npy file
np.save('preds_2.npy', preds_test1)







if __name__ == "__main__":
    # TODO: 调用你想执行的主函数，比如 main()
    pass
