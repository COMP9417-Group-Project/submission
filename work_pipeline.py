# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
<<<<<<< HEAD

import sys

=======
>>>>>>> 3a43ae6393ffca996cddcba6466e16cc18857673

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


# ============================== Preprocessing (Standardisation)========================

# define scaler
scaler = MinMaxScaler()
scaler.fit(X_train_df)

# transform data
X_train_scaled = scaler.transform(X_train_df)
X_test_scaled_labelled = scaler.transform(X_test_two_labelled)
X_test_scaled_unlabelled = scaler.transform(X_test_two_unlabelled)



# ============================== Helper Funtions========================
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


# ============================== Optimal Adaptation and Base Model Performance========================

# ### Applied the optimal generalised label shift adaptation to the base model and compare perfromance

def class_conditional_weights(source_X, source_y, target_X, target_y):
    """
        This implementation is based on the class-conditional importance weighting approach from:
        Bickel, S., Brückner, M., & Scheffer, T. (2009). 
        "Discriminative learning under covariate shift." 
        Journal of Machine Learning Research, 10, 2137-2155.
        
        The algorithm estimates importance weights separately for each class by training
        a discriminative model to distinguish between source and target domain instances.
    """
    source_y_np = np.array(source_y).flatten()
    target_y_np = np.array(target_y).flatten()
        
    # Get unique classes
    unique_classes = np.unique(source_y_np)
    weights = np.ones(len(source_X))
    
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
            lr = LogisticRegression(random_state=0, class_weight='balanced', max_iter=2000, C=0.7, penalty='l2')
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
    return weights

# without weights adaptive
# fit models
base_model = LogisticRegression(max_iter=2000, random_state=0)
base_model.fit(X_train_scaled, y_train)

# evaluate base model
print(f"="*50)
print(f"Resutls without weights adaptation")
base_lg_ce = evaluate_model(base_model, X_test_scaled_labelled, y_test_two_label.values.flatten())
base_lg_pred = base_model.predict(X_test_scaled_labelled)

# Resutls without weights adaptation
# Weighted CE: 0.02065
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.50      0.75      0.60         4
#            5       0.39      1.00      0.56         9
#            6       0.50      1.00      0.67         3
#            7       1.00      0.29      0.44        14
#            8       0.86      0.83      0.84        29
#            9       0.00      0.00      0.00         3
#           10       0.10      1.00      0.17         2
#           11       1.00      0.89      0.94         9
#           12       0.78      0.74      0.76        43
#           13       0.50      1.00      0.67         1
#           14       0.00      0.00      0.00         6
#           15       0.00      0.00      0.00         1
#           17       0.88      0.70      0.78        10
#           18       0.00      0.00      0.00         1
#           19       0.67      0.50      0.57         4
#           20       1.00      0.33      0.50         3
#           21       0.88      1.00      0.93         7
#           23       0.00      0.00      0.00         1
#           24       0.41      0.47      0.44        15
#           25       0.93      0.54      0.68        26
#           26       0.00      0.00      0.00         1
#           27       1.00      0.62      0.77         8

#     accuracy                           0.64       202
#    macro avg       0.47      0.49      0.43       202
# weighted avg       0.73      0.64      0.65       202

# # applied optimal adaptive weights model
# # get class conditional weights
weights = class_conditional_weights(
    source_X=X_train_scaled,  
    source_y=y_train.values.flatten(),
    target_X=X_test_scaled_labelled,  
    target_y=y_test_two_label.values.flatten(),
)

adpative_clf = LogisticRegression(random_state=0, max_iter=2000)
adpative_clf.fit(X_train_scaled, y_train, sample_weight=weights)


# evaluate adapative model
print(f"="*50)
print(f"Resutls with weights adaptation")
adaptive_lg_ce = evaluate_model(adpative_clf, X_test_scaled_labelled, y_test_two_label.values.flatten())
adaptive_lg_pred = adpative_clf.predict(X_test_scaled_labelled)
print(f"Weigthed CE gained: {base_lg_ce - adaptive_lg_ce:.4f}")

# Resutls with weights adaptation
# Weighted CE: 0.01646
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.80      1.00      0.89         4
#            5       0.43      1.00      0.60         9
#            6       0.50      1.00      0.67         3
#            7       1.00      0.50      0.67        14
#            8       0.90      0.93      0.92        29
#            9       0.00      0.00      0.00         3
#           10       0.14      1.00      0.25         2
#           11       1.00      0.89      0.94         9
#           12       0.81      0.81      0.81        43
#           13       0.50      1.00      0.67         1
#           14       0.00      0.00      0.00         6
#           15       0.00      0.00      0.00         1
#           17       1.00      0.70      0.82        10
#           18       0.00      0.00      0.00         1
#           19       1.00      0.75      0.86         4
#           20       0.67      0.67      0.67         3
#           21       1.00      1.00      1.00         7
#           23       1.00      1.00      1.00         1
#           24       0.52      0.73      0.61        15
#           25       1.00      0.62      0.76        26
#           26       0.00      0.00      0.00         1
#           27       1.00      0.75      0.86         8

#     accuracy                           0.74       202
#    macro avg       0.55      0.60      0.54       202
# weighted avg       0.79      0.74      0.74       202

# Weigthed CE gained: 0.0042



# ============================== Plots the performance improved per class with optimal weight adaptation strategy========================
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

def compare_class_performance(y_true, pred1, pred2, output_dir="plots/results"):
    """Compare per-class F1 scores between two models"""
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.title('F1 Score Improvement by Class (Adaptive Logistic Regression vs Baseline Logistic Regression)')
    plt.ylabel('F1 Score Difference')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/results/Class_conditional_weight_adpted_vs_baseline.png")
    plt.show()

compare_class_performance(y_test_two_label, base_lg_pred, adaptive_lg_pred)


# ### Discussion about weights adaptation
# Based on the above result, the weight adaptation method 
# improved performance is seen on the most of the classes indicating the weight adaptation 
# effectively migtigate the effect of the conditional shift


# ============================== Fine tuning the adapted models========================
# Compare with baseline model
print(f"="*50)
print(f"Fine tuned lg adapted model results")
trial_clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, C=0.7, penalty='l2')
trial_clf.fit(X_train_scaled, y_train, sample_weight=weights)

# evaluation
trial_lg_ce = evaluate_model(trial_clf, X_test_scaled_labelled, y_test_two_label.values.flatten())
trial_pred = trial_clf.predict(X_test_scaled_labelled)

print(f"Weigthed CE gained: {base_lg_ce - trial_lg_ce:.5f}")


# best ce logistic regression model
# trial_clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, C=0.7, penalty='l2')
# Weighted CE: 0.0123
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.57      1.00      0.73         4
#            5       1.00      0.89      0.94         9
#            6       0.75      1.00      0.86         3
#            7       0.91      0.71      0.80        14
#            8       0.86      0.83      0.84        29
#            9       0.50      0.33      0.40         3
#           10       0.67      1.00      0.80         2
#           11       0.90      1.00      0.95         9
#           12       0.83      0.67      0.74        43
#           13       0.11      1.00      0.20         1
#           14       1.00      1.00      1.00         6
#           15       1.00      1.00      1.00         1
#           17       0.90      0.90      0.90        10
#           18       0.00      0.00      0.00         1
#           19       0.50      0.50      0.50         4
#           20       0.67      0.67      0.67         3
#           21       1.00      1.00      1.00         7
#           23       0.25      1.00      0.40         1
#           24       0.59      0.67      0.62        15
#           25       0.90      0.69      0.78        26
#           26       0.00      0.00      0.00         1
#           27       1.00      0.88      0.93         8

#     accuracy                           0.76       202
#    macro avg       0.62      0.70      0.63       202
# weighted avg       0.82      0.76      0.78       202

# Weigthed CE gained: 0.0083


# best macro avg F1 logistic regression model

# trial_clf = LogisticRegression(random_state=0, class_weight='balanced', C=32, penalty='l2')
# Weighted CE: 0.0137
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

# Weigthed CE gained: 0.0069


# ### Trials on other models and fine tuning 
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import compute_sample_weight
from sklearn.ensemble import HistGradientBoostingClassifier


# define a simple svm model
simple_sgd = SGDClassifier(random_state=0, loss='log_loss')
simple_sgd.fit(X_train_scaled, y_train)
base_sgd_ce = evaluate_model(simple_sgd, X_test_scaled_labelled, y_test_two_label.values.flatten())


# # abdaptive svm
adpative_sgd = SGDClassifier(random_state=0, loss='log_loss', tol=1e-3 )
adpative_sgd.fit(X_train_scaled, y_train, sample_weight=weights)


# evaluate adapative model
print(f"="*50)
print(f"Resutls with weights adaptation(SGD)")
adaptive_sgd_ce = evaluate_model(adpative_sgd, X_test_scaled_labelled, y_test_two_label.values.flatten())
adaptive_sgd_pred = adpative_sgd.predict(X_test_scaled_labelled)

print(f"Weigthed CE gained: {base_sgd_ce - adaptive_sgd_ce:.4f}")

# ==================================================
# Resutls with weights adaptation
# Weighted CE: 0.02007
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       1.00      1.00      1.00         4
#            5       0.38      1.00      0.55         9
#            6       0.50      1.00      0.67         3
#            7       1.00      0.43      0.60        14
#            8       0.86      0.86      0.86        29
#            9       0.00      0.00      0.00         3
#           10       0.11      1.00      0.19         2
#           11       1.00      0.89      0.94         9
#           12       0.89      0.37      0.52        43
#           13       1.00      1.00      1.00         1
#           14       0.00      0.00      0.00         6
#           15       0.00      0.00      0.00         1
#           17       1.00      0.70      0.82        10
#           18       0.00      0.00      0.00         1
#           19       1.00      0.75      0.86         4
#           20       0.38      1.00      0.55         3
#           21       1.00      0.86      0.92         7
#           23       0.00      0.00      0.00         1
#           24       0.40      0.67      0.50        15
#           25       0.62      0.81      0.70        26
#           26       1.00      1.00      1.00         1
#           27       1.00      0.38      0.55         8

#     accuracy                           0.63       202
#    macro avg       0.55      0.57      0.51       202
# weighted avg       0.74      0.63      0.63       202

# Weigthed CE gained: 0.0057


# ============================== Class-specific Ensembles ========================
from sklearn.base import BaseEstimator, ClassifierMixin
# ### Ensemble on Adaptive Logistic Regression and SGD(Optimal models)

class ClassSpecificEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, sgd_model, lr_model, sgd_classes):
        self.sgd_model = sgd_model # fitted model
        self.lr_model = lr_model  # fitted model
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

# Define classes to prioritise with SGD
sgd_target_classes = [4, 13, 26] 

# Train models
best_lg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, C=0.7, penalty='l2')
best_lg.fit(X_train_scaled, y_train, sample_weight=weights)

best_sgd = SGDClassifier(random_state=0, loss='log_loss', tol=1e-3 )
best_sgd.fit(X_train_scaled, y_train, sample_weight=weights)


# Create Class-specific ensemble with fine tuned adapted models
ensemble = ClassSpecificEnsemble(
    sgd_model=best_sgd,
    lr_model=best_lg,
    sgd_classes=sgd_target_classes
)
ensemble.fit(X_train_scaled, y_train)
ensemble_ce = evaluate_model(ensemble, X_test_scaled_labelled, y_test_two_label.values.flatten())




# ============================== Semi-supervised Data Integration ========================

scores = {
    'baseline':[],
    'combined_train_test2_true_labelled': [],
    'combined_train_test2_pseudo_labelled':[],
    'combined_train_test2_pseudo_labelled(filtered)': [],
    'combined_train_test2_pseudo_n_true_labelled': []
}

# test set two pseudo labelling
pseudo_set2_labels = ensemble.predict(X_test_scaled_unlabelled)

# ============================== Data Preparation ==============================
# Split train/val
X_train_set1, X_val_set1, y_train_set1, y_val_set1 = train_test_split(
    X_train_df, y_train, test_size=0.2, random_state=0, stratify=y_train)

# Scale data
scaler = MinMaxScaler().fit(X_train_set1)
X_train_set1_scaled = scaler.transform(X_train_set1)
X_val_set1_scaled = scaler.transform(X_val_set1)
X_set2_scaled_unlabelled = scaler.transform(X_test_two_unlabelled)
X_set2_scaled_labelled = scaler.transform(X_test_two_labelled)

# ============================== Model Evaluation Function =====================
def train_and_evaluate_model(X_train, y_train, X_val, y_val, model):
    model.fit(X_train, y_train)
    return evaluate_model(model, X_val, y_val)

# ============================== Baseline Model =================================
set1_baseline = LogisticRegression(max_iter=2000, random_state=0)
scores['baseline'] = train_and_evaluate_model(
    X_train_set1_scaled, y_train_set1.values.flatten(),
    X_val_set1_scaled, y_val_set1.values.flatten(),
    set1_baseline
)

# ============================== Semi-supervised Data Integration Strategies Experiments ========================

def get_filtered_pseudo(X, threshold=0.4):
    proba = ensemble.predict_proba(X)
    confidence = np.max(proba, axis=1)
    mask = confidence >= threshold
    return X[mask], pseudo_set2_labels[mask]

# Strategy 1: Combined with true labelled test2
X_comb_true = np.concatenate([X_train_set1_scaled, X_set2_scaled_labelled], axis=0)
y_comb_true = np.concatenate([y_train_set1.values.flatten(), y_test_two_label.values.flatten()], axis=0)
scores['combined_train_test2_true_labelled'] = train_and_evaluate_model(
    X_comb_true, y_comb_true, X_val_set1_scaled, y_val_set1.values.flatten(),
    LogisticRegression(max_iter=2000, random_state=0)
)

# Strategy 2: Pseudo-labelled (unfiltered)
X_comb_pseudo = np.concatenate([X_train_set1_scaled, X_set2_scaled_unlabelled], axis=0)
y_comb_pseudo = np.concatenate([y_train_set1.values.flatten(), pseudo_set2_labels], axis=0)
scores['combined_train_test2_pseudo_labelled'] = train_and_evaluate_model(
    X_comb_pseudo, y_comb_pseudo, X_val_set1_scaled, y_val_set1.values.flatten(),
    LogisticRegression(max_iter=2000, random_state=0)
)
# best params: LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000, C=0.7, penalty='l2' )

# Strategy 3: Pseudo-labelled (filtered)
X_pseudo_filt, y_pseudo_filt = get_filtered_pseudo(X_set2_scaled_unlabelled)
X_comb_filt = np.concatenate([X_train_set1_scaled, X_pseudo_filt], axis=0)
y_comb_filt = np.concatenate([y_train_set1.values.flatten(), y_pseudo_filt], axis=0)
scores['combined_train_test2_pseudo_labelled(filtered)'] = train_and_evaluate_model(
    X_comb_filt, y_comb_filt, X_val_set1_scaled, y_val_set1.values.flatten(),
    LogisticRegression(max_iter=2000, random_state=0)
)

# Strategy 4: Combined pseudo + true labels
X_comb_all = np.concatenate([
    X_train_set1_scaled,
    X_set2_scaled_unlabelled,
    X_set2_scaled_labelled
], axis=0)
y_comb_all = np.concatenate([
    y_train_set1.values.flatten(),
    pseudo_set2_labels,
    y_test_two_label.values.flatten()
], axis=0)
scores['combined_train_test2_pseudo_n_true_labelled'] = train_and_evaluate_model(
    X_comb_all, y_comb_all, X_val_set1_scaled, y_val_set1.values.flatten(),
    LogisticRegression(max_iter=2000, random_state=0)
)




# ============================== Comparison Plot ================================
def plot_model_comparison(scores, output_dir='plots/results'):
    """
    Plot comparison of weighted CE Loss across different methods.
    
    Args:
        scores (dict): Dictionary mapping method names to their score values
        output_dir (str): Directory path to save the output figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    methods = list(scores.keys())
    values = [scores[m] for m in methods]

    bars = plt.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylabel('Weighted CE Loss', fontsize=12)
    plt.title('Model Comparison: Multi-source Training Data Integration', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(min(values)*0.95, max(values)*1.05)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_data_integration.png'))
    plt.show()

plot_model_comparison(scores)


# ============================== Fine tune with the best data integration strategy ================================
X_comb_pseudo = np.concatenate([X_train_set1_scaled, X_set2_scaled_unlabelled], axis=0)
y_comb_pseudo = np.concatenate([y_train_set1.values.flatten(), pseudo_set2_labels], axis=0)

train_and_evaluate_model(
    X_comb_pseudo, y_comb_pseudo, X_val_set1_scaled, y_val_set1.values.flatten(),
    LogisticRegression(random_state=0, class_weight='balanced', max_iter=2000, C=0.7, penalty='l2')
)

# Weighted CE: 0.00398
#   _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00         4
#            1       0.00      0.00      0.00         1
#            2       0.00      0.00      0.00         1
#            3       0.21      0.46      0.29        13
#            4       0.55      0.62      0.58        48
#            5       0.96      0.66      0.78       896
#            6       0.92      0.76      0.83       111
#            7       0.34      0.71      0.46        21
#            8       0.78      0.66      0.72       103
#            9       0.05      0.20      0.08         5
#           10       0.92      0.57      0.70       216
#           11       0.42      0.81      0.55        16
#           12       0.62      0.47      0.54        91
#           13       0.11      0.17      0.13        12
#           14       0.12      0.62      0.20        53
#           15       0.50      1.00      0.67         5
#           16       0.50      1.00      0.67         1
#           17       0.72      0.82      0.76        71
#           18       0.28      0.92      0.42        12
#           19       0.49      0.63      0.55        35
#           20       0.40      0.74      0.52        31
#           21       0.81      0.72      0.76        54
#           22       0.00      0.00      0.00         1
#           23       0.19      0.75      0.30         8
#           24       0.45      0.40      0.42        77
#           25       0.44      0.57      0.49        37
#           26       0.59      0.52      0.55        56
#           27       0.60      1.00      0.75        21

#     accuracy                           0.64      2000
#    macro avg       0.43      0.56      0.46      2000
# weighted avg       0.79      0.64      0.69      2000