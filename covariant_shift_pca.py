# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
y_train_path = '../../y_train.csv'
y_test_path = '../../y_test_2_reduced.csv'
x_train_path = '../../X_train.csv'
x_test_one_path = '../../X_test_1.csv'
x_test_two_path = '../../X_test_2.csv'


# read df
y_train = pd.read_csv(y_train_path)
y_test_two_label = pd.read_csv(y_test_path)
X_train_df = pd.read_csv(x_train_path)
X_test_one = pd.read_csv(x_test_one_path)
X_test_two = pd.read_csv(x_test_two_path)

# NOTE: using standard normalization
scaler = StandardScaler()
X_train_df = pd.DataFrame(scaler.fit_transform(X_train_df))
X_test_one = pd.DataFrame(scaler.transform(X_test_one))
X_test_two = pd.DataFrame(scaler.transform(X_test_two))

# split X_train_df into two part to get standard PCA reconstructed data
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train_df, y_train, test_size=0.3, random_state=42, stratify=y_train)


# **using nannyml detect shift and plot visualizations**



import nannyml as nml
from sklearn.decomposition import IncrementalPCA

feature_column_names = X_train_df.columns.tolist()

n_components = 50  
# sqrt(n_features)
# log2(n_features)

chunk_size = 500
# chunk_size='auto'

class HighDimReconstructor(nml.DataReconstructionDriftCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reconstructor = IncrementalPCA(n_components=n_components)

# init nml drift calculator
drift_calculator = HighDimReconstructor(
    column_names=feature_column_names,
    chunk_size=chunk_size,
    n_components=n_components
).fit(reference_data=X_train_df)

# analysis test set
drift_result1 = drift_calculator.calculate(data=X_test_one)

drift_result2 = drift_calculator.calculate(data=X_test_two)

fig = drift_result1.plot(
    kind='drift',
    plot_reference=True,
    metric='reconstruction_error',
    confidence_band_alpha=0.2
)
fig.update_layout(title="Drift Analysis of Test Set 1")
fig.show()

fig = drift_result2.plot(
    kind='drift',
    plot_reference=True,
    metric='reconstruction_error',
    confidence_band_alpha=0.2
)
fig.update_layout(title="Drift Analysis of Test Set 2")
fig.show()


significant_chunks = drift_result2.filter(
    period='analysis',
    drifted=True
)
print(f"Number of significant shifting chunks: {len(significant_chunks)}")

# output significant shifting features
if hasattr(drift_calculator._reconstructor, 'components_'):
    components = drift_calculator._reconstructor.components_
    feature_importance = np.sum(np.abs(components), axis=0)
    top10_features_idx = np.argsort(feature_importance)[-10:]
    top10_features = [feature_column_names[i] for i in top10_features_idx]
    print("TOP 10 Significant Shifting Features:", top10_features)


# Using the approach from nannyml to detect covariant shift in multivariate feature drift, using the PCA reconstruction Error as the metric.
# More details of the approach can be found in their blog page: https://www.nannyml.com/blog/detecting-covariate-shift-multivariate-approach

# ## Concept Distribution Shifting

# **Harmless concept shift**
# The Pred_proba has significant concept shift, but it is harmless. The model is still able to make accurate predictions on the test set, only the confidences vary.
# > Ideas From: https://www.nannyml.com/blog/concept-drift#concept-drift-detection-with-nannyml
# > Performance metrics such as accuracy, precision, recall, or f1-score will be the same for both models (ROC AUC will be impacted, though, since it uses the model scores rather than just class assignments). 
# 
# > concept shift might occur in any region within the feature space. If it happens to be in a sparse region, its impact on the model’s performance will be minor. This is because there is not much training nor serving data in this region. Thus, the model will hardly ever get to predict in this region. Any misclassifications caused by the concept shift in a sparse region will be rare events, not contributing much to the model’s overall performance.

# Use F1-score as evaluation metric, compare the performance on train and test set use SVM (baseline) as the classifier.



from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.manifold import TSNE
import plotly.express as px

# split the labelled test data
X_test_two_label = X_test_two[:202]

# Compute mean feature vector for each class
class_means = []
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_means.append(X_train_df[y_train['label'] == cls].mean(axis=0))
class_means = np.array(class_means)  # Shape: (n_classes, n_features)

n_class_clusters = 4
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
class_clusters = agg_clustering.fit_predict(class_means)
class_to_cluster = {cls: cluster for cls, cluster in zip(unique_classes, class_clusters)}

train_f1_scores = []
test_f1_scores = []

for cluster_id in range(n_class_clusters):
    cluster_classes = [cls for cls in unique_classes if class_to_cluster[cls] == cluster_id]

    mask_train = np.isin(y_train, cluster_classes)
    X_cluster_train = X_train_df[mask_train]
    y_cluster_train = y_train[mask_train]
    
    model = SVC(C=5)
    model.fit(X_cluster_train, y_cluster_train)

    y_train_pred = model.predict(X_cluster_train)
    train_f1 = f1_score(y_cluster_train, y_train_pred, average='weighted')
    train_f1_scores.append(train_f1)
    
    mask_test = np.isin(y_test_two_label, cluster_classes)
    X_cluster_test = X_test_two_label[mask_test]
    y_cluster_test = y_test_two_label[mask_test]

    if len(y_cluster_test) == 0:
        test_f1 = 0
    else:
        y_test_pred = model.predict(X_cluster_test)
        test_f1 = f1_score(y_cluster_test, y_test_pred, average='weighted')
    test_f1_scores.append(test_f1)

clusters = range(n_class_clusters)

# plot the F1 scores for each cluster
plt.figure(figsize=(12, 6))
plt.bar(clusters, train_f1_scores, width=0.4, label='Train F1', align='center')
plt.bar(clusters, test_f1_scores, width=0.4, label='Test F1', align='edge', alpha=0.7)
plt.xticks(clusters, labels=[f'Cluster {i}' for i in clusters])
plt.xlabel('Cluster')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison between Train and Test per Cluster')
plt.legend()
plt.show()

# plot the f1 score difference between train and test per cluster
f1_diff = np.array(train_f1_scores) - np.array(test_f1_scores)
plt.figure(figsize=(12, 6))
plt.bar(clusters, f1_diff, color='red', alpha=0.6)
plt.xticks(clusters, labels=[f'Cluster {i}' for i in clusters])
plt.xlabel('Cluster')
plt.ylabel('F1 Difference (Train - Test)')
plt.title('Concept Drift Indicator: F1 Score Differences')
plt.axhline(0, color='black', linestyle='--')
plt.show()


# ## label shift

# Use Jensen–Shannon divergence to detect label drifting instead of using the KL divergence. JS divergence is a more suitable measure of distance between two probability distributions than the KL divergence, since it has following properties: more symmetric (JSD(X || Y) = JSD(Y || X)), bounded, and robust in the presence of mixed shifts (covariates + concepts); KL divergence is only available when the distributions are strictly overlapping.



from scipy.spatial.distance import jensenshannon

n_classes = y_train['label'].unique().shape[0]

train_dist1 = np.bincount(y_train_1['label'], minlength=n_classes+1) / len(y_train_1)
train_dist2 = np.bincount(y_train_2['label'], minlength=n_classes+1) / len(y_train_2)
test_dist = np.bincount(y_test_two_label['label'], minlength=n_classes+1) / len(y_test_two_label)
js_divergence_train = jensenshannon(train_dist1, train_dist2, base=2)
js_divergence_test1 = jensenshannon(train_dist1, test_dist, base=2)
js_divergence_test2 = jensenshannon(train_dist2, test_dist, base=2)
print(f"Print Jensen-Shannon Divergence between Train Set (Part 1 & Part 2), Test Set 2:")
print(f"JSD Result Between y_train_1, y_train_2: {js_divergence_train:.4f}")
print(f"JSD Result Between y_train_1, y_test_2: {js_divergence_test1:.4f}")
print(f"JSD Result Between y_train_2, y_test_2: {js_divergence_test2:.4f}")


# ## Determine the Best K for clustering approach

# ### Sihouette Plot



from sklearn.metrics import silhouette_samples, silhouette_score

# calculate the cluster
silhouette_avg = silhouette_score(class_means, class_clusters)
sample_silhouette_values = silhouette_samples(class_means, class_clusters)

fig, ax = plt.subplots(figsize=(10, 8))
y_lower = 10

for i in range(n_class_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[class_clusters == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = plt.cm.tab10(i / n_class_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
    
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 1

ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_yticks([])
plt.title(f"Silhouette Plot (k={n_class_clusters}, Avg={silhouette_avg})")
plt.show()


# ## Generate pseudo labelling on Test set 1

# using the train set as source and test set 1 as target domain



from sklearn.semi_supervised import SelfTrainingClassifier

base_model = SVC(C=5, random_state=42, probability=True) # NOTE:the clf needs to be able to predict proba
self_training_model = SelfTrainingClassifier(base_model, threshold=0.85, verbose=True)
y_mixed = np.concatenate([y_train_1['label'], np.full(X_test_one.shape[0], -1)])
X_mixed = np.vstack([X_train_1, X_test_one])

self_training_model.fit(X_mixed, y_mixed)

y_pred = self_training_model.predict(X_train_2)

# evaluate performance

print(classification_report (y_train_2, y_pred))

y_pred_all = self_training_model.predict(X_mixed)
y_pseudo = y_pred_all[len(y_train_1):] # NOTE: pseudo labels in the test set 1
print("view the first 10 pseudo labels in the test set 1 (genrated by the SVM pseudo-labeling method):")
print(y_pseudo[:10])

