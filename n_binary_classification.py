
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
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score, precision_recall_curve, auc
import logging
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
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
y_test = pd.read_csv(y_test_path)
X_train_df = pd.read_csv(x_train_path)
X_test_one = pd.read_csv(x_test_one_path)
X_test_two = pd.read_csv(x_test_two_path)

test_length = y_test.shape[0]
test_indics = np.arange(test_length)
X_test = X_test_two.iloc[test_indics]



# split train and test
X_train, X_val, y_train, y_val = train_test_split(
    X_train_df, y, test_size=0.2, random_state=0, stratify=y)


# define scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# transform x train
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# transform X_test
X_test = scaler.transform(X_test)


# Determine the order of the classification
labels = y_train.values.flatten()

# frequency based order
label_frequency = Counter(labels)
sorted_labels = [label for label, _ in label_frequency.most_common()]
print(label_frequency)
# reverse frequency 
sorted_labels.reverse()
num_classes = len(sorted_labels)

# random order
# random.shuffle(sorted_labels)

# accuracy based order, using logistic regression as the base model to determine which class is the easiest one

# # Determine the order of the classification based on balanced metrics
# unique_labels = np.unique(labels)
# num_classes = len(unique_labels)

# # Dictionary to store per-class performance metrics
# class_performance = {}

# # Create a custom scorer - Average Precision (area under precision-recall curve)
# def average_precision_score(y_true, y_score):
#     precision, recall, _ = precision_recall_curve(y_true, y_score)
#     return auc(recall, precision)

# ap_scorer = make_scorer(average_precision_score, needs_proba=True)

# # For each class, create a binary classification problem
# for label in unique_labels:
#     # Create binary target (1 for this class, 0 for others)
#     binary_target = (labels == label).astype(int)
    
#     # Train a logistic regression model with balanced class weights
#     lr = SVC(max_iter=10000, class_weight='balanced', kernel='rbf')
    
#     # Use multiple scoring methods that work well with imbalanced data
#     balanced_acc_scores = cross_val_score(lr, X_train_scaled, binary_target, cv=5, 
#                                         scoring='balanced_accuracy')
#     f1_scores = cross_val_score(lr, X_train_scaled, binary_target, cv=5, 
#                               scoring='f1')
#     # ap_scores = cross_val_score(lr, X_train_scaled, binary_target, cv=5, 
#     #                           scoring=ap_scorer)
    
#     # Store average metrics for this class
#     class_performance[label] = {
#         'balanced_accuracy': np.mean(balanced_acc_scores),
#         'f1_score': np.mean(f1_scores),
#         # 'average_precision': np.mean(ap_scores)
#     }

# sorting_metric = 'f1_score'

# # Sort labels by the chosen metric (from highest to lowest)
# sorted_labels = sorted(class_performance.keys(), 
#                      key=lambda x: class_performance[x][sorting_metric], 
#                      reverse=True)

# print(f"Classes ordered by classification difficulty using {sorting_metric} (easiest first):")
# for label in sorted_labels:
#     metrics = class_performance[label]
#     print(f"Class {label}: Bal.Acc={metrics['balanced_accuracy']:.4f}, " 
#           f"F1={metrics['f1_score']:.4f}")

# sorted_labels =  [16, 22, 2, 1, 0, 9, 15, 23, 18, 23, 3, 11, 7, 27, 20, 19, 25, 4, 14, 21, 26, 17, 24, 12, 6, 8, 10, 5]
print(f"Predicting order:\n {sorted_labels}")




# the originial problem is doing multi-class classification (0-27 total 28 classes), 
# since it has distribution shift and severe class imablance problem
# so we turn it into a 27 binary classification model
# based on the labels frequency in the training set, we classify the most frequent class first,
# then use the rest of data points to classify the next label and so on.

# the initial feature values/labels contains all samples
cur_feature_holder = X_train_scaled
cur_label_holder = y_train

class_mapping = {i: sorted_labels[i] for i in range(num_classes)}
trained_classifier = []


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Force write to disk

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        FlushingFileHandler("log.txt"),  
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

CLASSIFIERS = [
    ('Logistic Regression', lambda: LogisticRegression(
        random_state=0,
        C=28,
        max_iter=2000,
        class_weight='balanced',
        penalty='l2'
    )),
    ('SVM', lambda: SVC(
        class_weight='balanced', 
        probability=True, 
        kernel = 'rbf',
        random_state=0
    )),
    ('Random Forest', lambda: RandomForestClassifier(
        class_weight='balanced', 
        n_estimators = 200,
        random_state=0
    ))
]
    
def weighted_log_loss(y_true, y_pred): 
    """
        Compute the weighted cross-entropy (log loss) given true labels and predicted labels
        Parameters:
        - y_true: (N,) true labels (-1 or 1)
        - y_pred: (N,) Predicted labels (-1 or 1)
        Returns:
        - Weighted log loss (scalar). 
    """   
    # Map -1 → 0 and 1 → 1 for the probability of class 1
    y_pred_prob = (y_pred + 1) / 2
    
    # Add small epsilon to avoid log(0) or log(1)
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    
    # Count samples in each class
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == -1)
    
    # Calculate weights inversely proportional to class frequencies
    w_pos = 1.0 / n_pos if n_pos > 0 else 0
    w_neg = 1.0 / n_neg if n_neg > 0 else 0
    
    # Normalize weights to sum to 1
    total_weight = w_pos + w_neg
    w_pos /= total_weight
    w_neg /= total_weight
    
    # Assign weights to samples based on their class
    sample_weights = np.where(y_true == 1, w_pos, w_neg)
    
    # Transform true labels from {-1, 1} to {0, 1} for the log loss formula
    y_true_binary = (y_true + 1) / 2  # Maps -1→0 and 1→1
    
    # Calculate binary cross entropy with sample weights
    log_loss = -np.mean(
        sample_weights * (
            y_true_binary * np.log(y_pred_prob) + 
            (1 - y_true_binary) * np.log(1 - y_pred_prob)
        )
    )
    return log_loss

loss_history = []  # Initialize list to track loss per class iteration

print(f"Predicting order: {sorted_labels}")
for index, current_class in enumerate(sorted_labels):
    if len(cur_label_holder) == 0:
        print("No samples remaining - stopping iteration")
        break
        
    # # save the training data for the classified class (not encoded)
    # pd.DataFrame(cur_feature_holder, columns=X_train_df.columns).to_csv(
    #     f'x_train_split_{current_class}.csv', 
    #     index=False
    # )
    # cur_label_holder.to_csv(
    #     f'y_train_split_{current_class}.csv', 
    #     index=False
    # )
    
    print(f"\n{'='*40}")
    print(f"Processing class {current_class} ({index+1}/{num_classes})")

    unique_before, counts_before = np.unique(cur_label_holder.values, return_counts=True)
    print(f"Current label distribution: {dict(zip(unique_before, counts_before))}")
    
    # Before processing
    current_class_count = (cur_label_holder.values.flatten() == current_class).sum()
    other_class_count = len(cur_label_holder) - current_class_count


    # encode the y label of other classes into -1
    # iterate through the y labels
    # Create binary classification labels
    train_mask = (cur_label_holder.values.flatten() == current_class)
    
    positive_samples = train_mask.sum()
    negative_samples = len(train_mask) - positive_samples

    
    if positive_samples == 0:
        print(f"WARNING: No positive samples for {current_class} - skipping")
        continue
        
    if negative_samples == 0:
        print(f"WARNING: No negative samples for {current_class} - final class remains")
        # Assign all remaining samples to this class
        final_class = current_class
        break  # Exit loop as no more classes to process
        
    encoded_y_train = np.where(train_mask, 1, -1)  # Use 1/-1 for class distinction
    
    # Apply oversampling if positive class has fewer samples than negative class
    # if positive_samples < negative_samples:
    #     # print(f"Applying SMOTE oversampling for class {current_class}")
    #     # Convert features and labels for SMOTE
    #     feature_array = cur_feature_holder
    #     label_array = encoded_y_train
        
    #     # Initialize and apply SMOTE
    #     smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)
    #     feature_resampled, label_resampled = smote.fit_resample(feature_array, label_array)
        
    #     # Update counts after oversampling
    #     positive_samples_after = (label_resampled == 1).sum()
    #     negative_samples_after = (label_resampled == -1).sum()
        

    #     # Use oversampled data for training
    #     training_features = feature_resampled
    #     training_labels = label_resampled
    #     logger.info(f"Applied SMOTE oversampling for class {current_class}: {positive_samples} → {positive_samples_after}")
    # else:
    #     print(f"No oversampling needed for class {current_class} (positive samples >= negative samples)")
    #     training_features = cur_feature_holder
    #     training_labels = encoded_y_train
    
    val_mask = (y_val.values.flatten() == current_class)
    encoded_y_val = np.where(val_mask, 1, -1)

    # print(f"\nValidation set:")
    # print(f"Current class samples ({current_class}): {val_mask.sum()}")
    # print(f"Other classes samples: {len(y_val) - val_mask.sum()}")

    # ========== CLASSIFIER SELECTION BLOCK ==========
    best_clf = None
    best_clf_name = ""
    best_loss = 1000
    best_score = -1

   
    # Evaluate all classifiers
    for clf_name, clf_factory in CLASSIFIERS:
        # Create new classifier instance
        clf = clf_factory()
    
        # Original training process for other classifiers
        clf.fit(cur_feature_holder, encoded_y_train)
        val_pred = clf.predict(X_val_scaled)
        
        # Calculate F1 score for positive class
        score = f1_score(encoded_y_val, val_pred, pos_label=1)
        loss = weighted_log_loss(encoded_y_val, val_pred)
    
        
        # Update best classifier
        if score > best_score:
            best_score = score
        #     best_clf = clf
        #     best_clf_name = clf_name
    
        if loss < best_loss:
            best_loss = loss
            best_clf = clf
            best_clf_name = clf_name
            
    loss_history.append((current_class, best_loss))  # Track the best loss for this class
    print(f"Selected {best_clf_name} for {current_class} (F1: {best_score:.3f}) (Loss: {best_loss})")
    logger.info(f"Selected {best_clf_name} for {current_class} (F1: {best_score:.3f}) (Loss: {best_loss})")
    
    # ========== END CLASSIFIER SELECTION ==========
    
    # Use the best classifier for final evaluation
    val_pred = best_clf.predict(X_val_scaled)
    report = classification_report(encoded_y_val, val_pred)
    
    print(f"Validation scores for {current_class}:")
    print(report)
    logger.info(f"Classification Report for {current_class}:\n{report}")
    # Store both classifier and its name
    trained_classifier.append((best_clf_name, best_clf))
    
    # # define a classifier 
    # clf = LogisticRegression(random_state=0, solver='saga', class_weight='balanced')

    # # train with the classifier
    # clf.fit(cur_feature_holder, encoded_y_train)

    # # predict with the validation set
    # val_pred = clf.predict(X_val_scaled)
    # report = classification_report(encoded_y_val, val_pred)
    
    # print(f"Validation scores for {current_class}:")
    # print(report)
    # logger.info(f"Classification Report for {current_class}:\n{report}")
    
    # trained_classifier.append(clf)
    

    # remove the current classified class from the training sample
    remaining_mask = ~train_mask  # Invert mask to keep other classes
    remaining_class_count = remaining_mask.sum()
    
    # print(f"\nAfter removing {current_class}:")
    # print(f"Remaining samples: {remaining_class_count}")
    # print(f"Removed samples: {len(cur_label_holder) - remaining_class_count}")

    
    # Verify no current class remains
    remaining_current = (cur_label_holder[remaining_mask].values.flatten() == current_class).sum()
    # print(f"Verification - remaining {current_class} samples: {remaining_current}")

    # Update holders
    cur_feature_holder = cur_feature_holder[remaining_mask]
    cur_label_holder = cur_label_holder.iloc[remaining_mask]
    
    # Final check
    # print(f"\nUpdated feature holder shape: {cur_feature_holder.shape}")
    # print(f"Updated label holder count: {len(cur_label_holder)}")
    print('='*40)
    
# After processing all classes
if loss_history:
    plt.figure(figsize=(12, 6))
    
    # Create positions based on classification order
    x_positions = np.arange(1, len(loss_history) + 1)
    losses = [loss for _, loss in loss_history]
    classes = [cls for cls, _ in loss_history]
    
    # Plot main line
    plt.plot(x_positions, losses, marker='o', linestyle='-', color='b', markersize=8)
    
    # Add labels and annotations
    plt.xlabel('Classification Order', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss by Classification Order', fontsize=14)
    plt.xticks(x_positions, [f"{i+1}" for i in range(len(loss_history))], fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations with class and loss
    for idx, (cls, loss) in enumerate(loss_history):
        plt.text(x_positions[idx], 
                 loss + 0.005,  # Slight vertical offset
                 f'{cls}\n({loss:.3f})',
                 ha='center', 
                 va='bottom',
                 fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.show()
else:
    print("No loss data to plot.")




# inference
# make inference of each datasample in the test at a time

# print(X_test.shape)
# print(class_mapping)

def weighted_log_loss_test(y_true_np, y_pred):
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

    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

    # Calculate log loss term
    log_terms = np.sum(y_true_one_hot * np.log(y_pred), axis=1)
    loss = -np.mean(sample_weights * log_terms)
    
    return loss



# def evaluate_model(model, X, y_true):
#     """
#         model: fitted model
#         X: feature matrix (n_sample, 300)
#         y_true: array, contains true labels (0-27), (n_samples, )
#     """
#     y_pred_prob = model.predict_proba(X)  # (n_samples, 28)
#     y_pred_label = model.predict(X)
#     ce = weighted_log_loss(y_true, y_pred_prob)
#     print(f"Weighted CE: {ce:.4f}")
#     print(classification_report(y_true, y_pred_label))
#     return ce

# # iterate through all trained classifier
# # if the classifer return 1, then assign the true class label for that prediction
# test_pred = []
# for index, datapoint in enumerate(X_val_scaled):
#     datapoint = datapoint.reshape(1, -1)
#     # print(datapoint.shape)
#     for i, (classifier_name, classfier) in enumerate(trained_classifier):
#         pred = classfier.predict(datapoint)[0]
#         # if the current classifier is the last classifier, no matter what it is, use the prediction
#         if i == 26:
#             if pred == 1:
#                 pred_true = class_mapping[i]
#                 test_pred.append(pred_true)
#             else:
#                 pred_true = class_mapping[i+1]
#                 test_pred.append(pred_true)
#             break
            
#         # if current classifier is not the last classifier, then return the first true label
#         if pred == 1:
#             pred_true = class_mapping[i]
#             test_pred.append(pred_true)
#             break

# print(f"Cross Entropy score: {weighted_log_loss_test(y_val, test_pred):.4f}")
# print(classification_report(y_val, test_pred))

def get_class_probabilities(X, classifiers, class_mapping, sorted_labels):
    """
    Generate probability matrix [n_samples, 28] using cascaded classifiers.
    
    Args:
        X: Input features (n_samples, n_features)
        classifiers: List of trained classifiers
        class_mapping: Dictionary {order: original_class}
        sorted_labels: List of classes in classification order
    
    Returns:
        prob_matrix: (n_samples, 28) probability matrix
    """
    num_classes = len(class_mapping)
    prob_matrix = np.zeros((X.shape[0], num_classes))
    
    # Get unprocessed classes (if training stopped early)
    num_trained = len(classifiers)
    unprocessed_classes = sorted_labels[num_trained:]  # Classes after last trained classifier
    
    for idx, x in enumerate(X):
        x_reshaped = x.reshape(1, -1)
        remaining_prob = 1.0
        
        # Iterate through trained classifiers
        for i, (_, clf) in enumerate(classifiers):
            # Get the actual class this classifier corresponds to
            original_class = class_mapping[i]
            
            pos_prob = clf.predict_proba(x_reshaped)[0][1]
            
            # Assign to values in probability matrix
            prob_matrix[idx, original_class] = remaining_prob * pos_prob
            remaining_prob *= (1 - pos_prob)
        
        # Handle residual probability for unprocessed classes
        if unprocessed_classes:
            residual_prob = remaining_prob / len(unprocessed_classes)
            for cls in unprocessed_classes:
                prob_matrix[idx, cls] = residual_prob
    
    return prob_matrix

# During validation
val_probs = get_class_probabilities(
    X_val_scaled, 
    trained_classifier, 
    class_mapping, 
    sorted_labels
)

print(val_probs.shape)

# For evaluation
# Evaluate
print(f"Weighted Log Loss: {weighted_log_loss_test(y_val.values.flatten(), val_probs):.4f}")
print(classification_report(y_val, np.argmax(val_probs, axis=1)))




# reverse order
# Cross Entropy score: 0.034
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.43      0.75      0.55         4
#            5       0.50      1.00      0.67         9
#            6       0.40      0.67      0.50         3
#            7       0.89      0.57      0.70        14
#            8       0.88      0.76      0.81        29
#            9       0.00      0.00      0.00         3
#           10       0.20      1.00      0.33         2
#           11       0.88      0.78      0.82         9
#           12       0.83      0.56      0.67        43
#           13       0.00      0.00      0.00         1
#           14       0.50      0.17      0.25         6
#           15       0.00      0.00      0.00         1
#           17       0.88      0.70      0.78        10
#           18       0.00      0.00      0.00         1
#           19       0.50      0.50      0.50         4
#           20       0.10      0.33      0.15         3
#           21       0.88      1.00      0.93         7
#           23       0.00      0.00      0.00         1
#           24       0.38      0.53      0.44        15
#           25       0.73      0.62      0.67        26
#           26       0.12      1.00      0.22         1
#           27       1.00      0.88      0.93         8

#     accuracy                           0.63       202
#    macro avg       0.42      0.49      0.41       202
# weighted avg       0.70      0.63      0.64       202



# Predicting order: [15, 7, 24, 14, 17, 6, 23, 11, 8, 3, 18, 26, 27, 10, 25, 21, 1, 13, 12, 4, 22, 16, 20, 0, 19, 2, 9, 5]

# predicting order
# Cross Entropy score: 0.035
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.25      0.25      0.25         4
#            5       0.47      1.00      0.64         9
#            6       0.40      0.67      0.50         3
#            7       0.89      0.57      0.70        14
#            8       0.88      0.72      0.79        29
#            9       0.00      0.00      0.00         3
#           10       0.13      1.00      0.24         2
#           11       0.88      0.78      0.82         9
#           12       0.65      0.56      0.60        43
#           13       0.33      1.00      0.50         1
#           14       1.00      0.17      0.29         6
#           15       0.00      0.00      0.00         1
#           17       0.88      0.70      0.78        10
#           18       0.00      0.00      0.00         1
#           19       0.00      0.00      0.00         4
#           20       0.67      0.67      0.67         3
#           21       0.75      0.86      0.80         7
#           23       0.00      0.00      0.00         1
#           24       0.31      0.60      0.41        15
#           25       0.81      0.50      0.62        26
#           26       0.17      1.00      0.29         1
#           27       1.00      0.62      0.77         8

#     accuracy                           0.59       202
#    macro avg       0.44      0.49      0.40       202
# weighted avg       0.68      0.59      0.60       202

# Predicting order: [5, 2, 14, 26, 1, 11, 18, 12, 21, 23, 22, 16, 15, 17, 3, 4, 24, 13, 0, 6, 8, 25, 27, 19, 9, 7, 10, 20]
# Cross Entropy score: 0.034
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.36      1.00      0.53         4
#            5       0.39      1.00      0.56         9
#            6       0.40      0.67      0.50         3
#            7       0.88      0.50      0.64        14
#            8       0.92      0.76      0.83        29
#            9       0.00      0.00      0.00         3
#           10       0.18      1.00      0.31         2
#           11       0.80      0.89      0.84         9
#           12       0.68      0.70      0.69        43
#           13       0.00      0.00      0.00         1
#           14       0.00      0.00      0.00         6
#           15       1.00      1.00      1.00         1
#           17       0.78      0.70      0.74        10
#           18       0.00      0.00      0.00         1
#           19       0.50      0.25      0.33         4
#           20       1.00      0.67      0.80         3
#           21       1.00      0.71      0.83         7
#           23       0.00      0.00      0.00         1
#           24       0.44      0.53      0.48        15
#           25       0.75      0.46      0.57        26
#           26       0.10      1.00      0.18         1
#           27       1.00      0.25      0.40         8

#     accuracy                           0.61       202
#    macro avg       0.47      0.50      0.43       202
# weighted avg       0.68      0.61      0.61       202

# Predicting order: [9, 14, 2, 16, 0, 4, 12, 13, 19, 21, 26, 22, 25, 15, 24, 10, 11, 3, 18, 1, 6, 23, 7, 20, 27, 5, 8, 17]

# Cross Entropy score: 0.038
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.50      1.00      0.67         4
#            5       0.41      1.00      0.58         9
#            6       0.50      1.00      0.67         3
#            7       1.00      0.07      0.13        14
#            8       0.78      0.62      0.69        29
#            9       0.00      0.00      0.00         3
#           10       0.10      1.00      0.18         2
#           11       1.00      0.44      0.62         9
#           12       0.65      0.79      0.72        43
#           13       0.00      0.00      0.00         1
#           14       1.00      0.17      0.29         6
#           15       0.00      0.00      0.00         1
#           17       1.00      0.70      0.82        10
#           18       0.00      0.00      0.00         1
#           19       1.00      0.75      0.86         4
#           20       0.33      0.33      0.33         3
#           21       0.86      0.86      0.86         7
#           23       0.00      0.00      0.00         1
#           24       0.44      0.53      0.48        15
#           25       0.79      0.42      0.55        26
#           26       0.00      0.00      0.00         1
#           27       1.00      0.62      0.77         8

#     accuracy                           0.58       202
#    macro avg       0.47      0.43      0.38       202
# weighted avg       0.71      0.58      0.58       202

# Predicting order: [6, 16, 17, 19, 7, 22, 9, 5, 26, 11, 8, 27, 25, 23, 2, 20, 24, 3, 1, 18, 13, 21, 15, 10, 14, 4, 0, 12]
# Cross Entropy score: 0.031
#               precision    recall  f1-score   support

#            2       0.00      0.00      0.00         1
#            3       0.00      0.00      0.00         1
#            4       0.00      0.00      0.00         4
#            5       0.50      1.00      0.67         9
#            6       0.50      1.00      0.67         3
#            7       0.88      0.50      0.64        14
#            8       0.82      0.79      0.81        29
#            9       0.00      0.00      0.00         3
#           10       0.13      1.00      0.24         2
#           11       0.89      0.89      0.89         9
#           12       0.81      0.49      0.61        43
#           13       0.25      1.00      0.40         1
#           14       0.00      0.00      0.00         6
#           15       1.00      1.00      1.00         1
#           17       0.78      0.70      0.74        10
#           18       0.00      0.00      0.00         1
#           19       0.60      0.75      0.67         4
#           20       0.17      0.33      0.22         3
#           21       0.75      0.86      0.80         7
#           23       0.00      0.00      0.00         1
#           24       0.37      0.47      0.41        15
#           25       0.70      0.62      0.65        26
#           26       0.17      1.00      0.29         1
#           27       1.00      0.50      0.67         8

#     accuracy                           0.59       202
#    macro avg       0.43      0.54      0.43       202
# weighted avg       0.66      0.59      0.60       202






# rs = 0 
# # Predicting order:
#  [16, 22, 2, 1, 0, 9, 15, 23, 18, 13, 3, 11, 7, 27, 20, 19, 25, 4, 14, 21, 26, 17, 24, 12, 8, 6, 10, 5]
# Cross Entropy score: 0.0069
#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00         4
#            1       0.33      1.00      0.50         1
#            2       0.00      0.00      0.00         1
#            3       0.28      0.62      0.38        13
#            4       0.39      0.50      0.44        48
#            5       0.94      0.65      0.77       896
#            6       0.81      0.57      0.67       111
#            7       0.38      0.71      0.50        21
#            8       0.76      0.31      0.44       103
#            9       0.03      0.20      0.05         5
#           10       0.92      0.45      0.61       216
#           11       0.56      0.94      0.70        16
#           12       0.49      0.21      0.29        91
#           13       0.07      0.25      0.11        12
#           14       0.10      0.58      0.17        53
#           15       0.80      0.80      0.80         5
#           16       1.00      1.00      1.00         1
#           17       0.58      0.72      0.64        71
#           18       0.43      0.75      0.55        12
#           19       0.31      0.54      0.39        35
#           20       0.31      0.74      0.43        31
#           21       0.69      0.65      0.67        54
#           22       0.00      0.00      0.00         1
#           23       0.25      0.50      0.33         8
#           24       0.32      0.38      0.35        77
#           25       0.31      0.43      0.36        37
#           26       0.37      0.46      0.41        56
#           27       0.57      0.95      0.71        21

#     accuracy                           0.56      2000
#    macro avg       0.43      0.53      0.44      2000
# weighted avg       0.74      0.56      0.61      2000


# rs = 42
# Cross Entropy score: 0.0075
#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00         4
#            1       0.00      0.00      0.00         1
#            2       0.00      0.00      0.00         1
#            3       0.15      0.46      0.23        13
#            4       0.36      0.54      0.43        48
#            5       0.93      0.64      0.76       896
#            6       0.90      0.87      0.89       111
#            7       0.38      0.81      0.52        21
#            8       0.83      0.38      0.52       103
#            9       0.00      0.00      0.00         5
#           10       0.88      0.40      0.55       216
#           11       0.46      0.69      0.55        16
#           12       0.44      0.23      0.30        91
#           13       0.04      0.08      0.06        12
#           14       0.09      0.53      0.16        53
#           15       0.83      1.00      0.91         5
#           16       1.00      1.00      1.00         1
#           17       0.65      0.77      0.71        71
#           18       0.31      0.67      0.42        12
#           19       0.31      0.57      0.40        35
#           20       0.24      0.61      0.35        31
#           21       0.73      0.69      0.70        54
#           22       0.00      0.00      0.00         1
#           23       0.64      0.88      0.74         8
#           24       0.28      0.40      0.33        77
#           25       0.23      0.32      0.27        37
#           26       0.40      0.41      0.41        56
#           27       0.64      0.86      0.73        21

#     accuracy                           0.57      2000
#    macro avg       0.42      0.49      0.43      2000
# weighted avg       0.74      0.57      0.62      2000














if __name__ == "__main__":
    # TODO: 调用你想执行的主函数，比如 main()
    pass
