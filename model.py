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

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

