"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    
    # class proportions
    classes,p = np.unique(labels, return_counts=True)
    p = p / len(labels)
    
    # gini
    return 1 - np.sum(p ** 2)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # partition rows into left (feature <= threshold) and right (feature > threshold)
    mask = features[:, feature_index] <= threshold

    return features[mask], labels[mask], features[~mask], labels[~mask]

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # return a score where higher means the children are purer than the parent.
    
    return impurity(parent_labels) - (len(left_labels) * impurity(left_labels) + len(right_labels) * impurity(right_labels)) / len(parent_labels)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # search feature_indices for the (feature, threshold) that best improves purity.
    best = {
        'feature_index': None,
        'threshold': None,
        'score': 0.0
    }

    for fi in feature_indices:
        feats = np.sort(features[:, fi])
        t_candidates = (feats[:-1] + feats[1:]) / 2

        for t in t_candidates:
            lf, ll, rf, rl = split_dataset(features, labels, fi, t)

            # Skip cases where either side is empty
            if len(lf) == 0 or len(rf) == 0:
                continue

            s = split_score(labels, ll, rl)

            if s > best['score']:
                best = {
                    'feature_index': fi,
                    'threshold': t,
                    'score': s                    
                }

    return best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # decide whether to stop growing based on purity, depth, and size...
    return (len(np.unique(labels)) == 1) or (depth >= max_depth) or (len(labels) < min_samples_split)

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # choose a single class label to output for a leaf given the labels that reached it
    labels = np.array(labels, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    return int(classes[np.argmax(counts)])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {
            'leaf': True, 
            'prediction': leaf_prediction(labels)
        }

    if feature_subset is None:
        feature_subset = range(features.shape[1])

    best = best_split(features, labels, feature_subset)

    if best['feature_index'] is None:
        return {
            'leaf': True, 
            'prediction': leaf_prediction(labels)
        }

    lf, ll, rf, rl = split_dataset(features, labels, best['feature_index'], best['threshold'])        

    if len(lf) == 0 or len(rf) == 0:
        return {
            'leaf': True, 
            'prediction': leaf_prediction(labels)
        }

    return {
        'leaf': False,
        'feature_index': best['feature_index'],
        'threshold': best['threshold'],
        'left': build_tree(lf, ll, max_depth, min_samples_split, feature_subset, depth),
        'right': build_tree(rf, rl, max_depth, min_samples_split, feature_subset, depth)
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if tree['leaf']:
        return tree['prediction']

    return predict_example_tree(tree['left' if (example[tree['feature_index']] <= tree['threshold']) else 'right'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # return predicted class for each row of features using the fitted tree.
    return np.array([predict_example_tree(tree, row) for row in features])

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # draw a bootstrap sample of rows (with replacement) using rng.
    indices = rng.integers(0, features.shape[0], size=features.shape[0])
    return features[indices], labels[indices]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # return num_to_pick distinct random feature indices from range(num_features) using rng.
    if num_to_pick >= 1 and num_to_pick <= num_features:
        return rng.choice(num_features, size=num_to_pick, replace=False)
    else:
        return np.array([], dtype=int)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    # grow num_trees decision trees on bootstrap samples with random feature subsets.
    num_features = features.shape[1]
    if num_features_per_split is None:
        num_features_per_split = round(float(np.sqrt(num_features)))

    rng =  np.random.default_rng(random_state)

    forest = []

    for _ in range(num_trees):
        feature_indices =  feature_subset(num_features, num_features_per_split, rng)
        f_sample, l_sample = bootstrap_sample(features, labels, rng)

        forest.append({
            'tree': build_tree(f_sample, l_sample, max_depth, min_samples_split, feature_indices, depth=0),
            'feature_indices': feature_indices
        })

    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    # aggregate the per-tree predictions of an ensemble into one prediction per example.
    
    tree_predictions = np.array(tree_predictions)
    n = tree_predictions.shape[1]
    preds = np.zeros(n)

    for i in range(n):
        classes, freqs = np.unique(tree_predictions[:, i], return_counts=True)
        preds[i] = int(classes[np.argmax(freqs)])

    return preds

# Step 14 - predict_forest
def predict_forest(forest, features):
    # predict classes for a dataset using the whole trained forest.
    
    return combine_predictions(np.stack([predict_tree(entry['tree'], features) for entry in forest]))

# Step 15 - accuracy
def accuracy(predictions, labels):
    # compute the fraction of entries where predictions equals labels
    return np.sum(predictions == labels) / len(labels)

