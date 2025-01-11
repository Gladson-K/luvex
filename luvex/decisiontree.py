class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
       
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        #assert 0 <= self.feature < X_test.shape[1], f"Invalid feature index: {node.feature}"

    def is_leaf_node(self):
        return self.value is not None

import numpy as np
from collections import Counter

class DecisionTree():
    #initializing stopping criteria values
    def __init__(self,min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
        
    def fit(self,X,y):

        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)
        

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

    # Stop criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

    # Find the best split
        best_feature, best_thresh = self._best_split(X, y)
        if best_feature is None or best_thresh is None:  # No valid split
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

    # Split data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)
        
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None

        best_gain = -float("inf")
        split_feature, split_threshold = None, None

        for feature in range(n_features):
            X_column = X[:, feature]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold

        #print(f"Best feature: {split_feature}, Best threshold: {split_threshold}, Best gain: {best_gain}")
        return split_feature, split_threshold

                

    def _information_gain(self,y,X_col,thr):     

        #get the parent entropy
        parent_ent = self._entropy(y)

        #create child Node
        left_idx, right_idx = self._split(X_col,thr)

        if len(left_idx)==0 or len(right_idx)==0:
            return 0

        #calculate the weighted entropy of child Node
        n = len(y)
        n_le, n_ri = len(left_idx),len(right_idx)
        e_le , e_ri = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        #print(f"n_le: {n_le}, n_ri: {n_ri}, e_le: {e_le}, e_ri: {e_ri}, n: {n}")

        child_ent = (n_le/n)*(e_le)+(n_ri/n)*(e_ri) # weighted average
        
        #calculate the Information Gain

        inform_gain = parent_ent - child_ent

        return inform_gain

    def _split(self,X_col,thr):
        left_idxs = np.argwhere(X_col<=thr).flatten() #indices of array element which are non-zero
        right_idxs = np.argwhere(X_col>thr).flatten()
        return left_idxs, right_idxs
        
    def _entropy(self, y):
        hist = np.bincount(y) #from 0 to y it count no of times hist is present
        p_x = hist/len(y) 
        return -np.sum([p*np.log2(p) for p in p_x if p>0])
        
    def _most_common_label(self,y):
        
        #Return the most common label by using Counter class
        
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _traverse_tree(self, x, node):

        #print(f"x[node.feature]: {x[node.feature]}, node.threshold: {node.threshold}")

        if node.is_leaf_node():
            return node.value
            
        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
        
    def predict(self, X):

        return np.array([self._traverse_tree(x, self.root) for x in X])