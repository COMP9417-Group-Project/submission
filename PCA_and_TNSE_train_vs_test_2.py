import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Load the datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test_2 = pd.read_csv('X_test_2.csv').head(202)
y_test_2_reduced = pd.read_csv('y_test_2_reduced.csv')

# Get unique class labels
unique_classes = sorted(y_train['label'].unique())

# Set PCA parameters
n_components = 2
random_state = 42

for class_label in unique_classes:
    # Filter data for the current class
    train_indices = y_train[y_train['label'] == class_label].index
    X_train_class = X_train.loc[train_indices]

    test_indices = y_test_2_reduced[y_test_2_reduced['label'] == class_label].index
    X_test_2_class = X_test_2.loc[test_indices]

    if X_train_class.shape[0] > 0 and X_test_2_class.shape[0] > 0:
        # Combine features and create domain labels
        X_combined = pd.concat([X_train_class, X_test_2_class], ignore_index=True)
        domain_labels = np.concatenate([np.zeros(X_train_class.shape[0]), np.ones(X_test_2_class.shape[0])])

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        principal_components = pca.fit_transform(X_combined)

        # Separate principal components for train and test
        train_pca = principal_components[:X_train_class.shape[0]]
        test_pca = principal_components[X_train_class.shape[0]:]

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(train_pca[:, 0], train_pca[:, 1], label='Train', alpha=0.7)
        plt.scatter(test_pca[:, 0], test_pca[:, 1], label='Test_2', alpha=0.7)
        plt.title(f'PCA Visualization - Class: {class_label}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()

    else:
        print(f"Warning: Class {class_label} not present in both training and test_2 data. Skipping PCA.")

print("PCA plots display process completed.")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Load the datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test_2 = pd.read_csv('X_test_2.csv').head(202)
y_test_2_reduced = pd.read_csv('y_test_2_reduced.csv')

# Get unique class labels
unique_classes = sorted(y_train['label'].unique())

# Set t-SNE parameters
n_components = 2  # For 2D visualization
perplexity = 30
random_state = 42

for class_label in unique_classes:
    # Filter data for the current class
    train_indices = y_train[y_train['label'] == class_label].index
    X_train_class = X_train.loc[train_indices]

    test_indices = y_test_2_reduced[y_test_2_reduced['label'] == class_label].index
    X_test_2_class = X_test_2.loc[test_indices]

    if X_train_class.shape[0] > 0 and X_test_2_class.shape[0] > 0:
        # Combine features and create domain labels
        X_combined = pd.concat([X_train_class, X_test_2_class], ignore_index=True)
        domain_labels = np.concatenate([np.zeros(X_train_class.shape[0]), np.ones(X_test_2_class.shape[0])])

        # Apply t-SNE only if the number of samples is greater than perplexity
        if X_combined.shape[0] > perplexity:
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            embeddings = tsne.fit_transform(X_combined)

            # Separate embeddings for train and test
            train_embeddings = embeddings[:X_train_class.shape[0]]
            test_embeddings = embeddings[X_train_class.shape[0]:]

            # Create the scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], label='Train', alpha=0.7)
            plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], label='Test_2', alpha=0.7)
            plt.title(f't-SNE Visualization - Class: {class_label}')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Show the plot
            plt.show()

        else:
            print(f"Warning: Insufficient samples ({X_combined.shape[0]}) for perplexity in Class {class_label}. Skipping t-SNE.")

    else:
        print(f"Warning: Class {class_label} not present in both training and test_2 data. Skipping.")

print("t-SNE plots display process completed.")


