# Decision Tree Model
## Introduction
A decision tree algorithm is a popular and intuitive method used in machine learning for classification and regression tasks. It works by splitting a dataset into subsets based on the value of input features, forming a tree-like structure. Each node in the tree represents a decision point based on the value of an attribute, and each branch represents the outcome of the decision, leading to either another decision node or a leaf node, which represents a final output or class label.

Installing basic libraries needed for the `Decision Tree Model` and `Dataset Collection`
```bash
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install scipy
pip install scikit-learn
pip install plotly
pip install ucimlrepo
``` 

## Dataset
<a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank">
    <button style='background-color: #1b1c18;
        border: 1px solid white;
        border-radius: 10px;
        color: white;
        padding: 10px 15px;
        text-align: center;
        text-decoration: none;
        font-size: 14px;
        margin-bottom: 1rem;
        cursor: pointer;'>
        Breast Cancer Dataset
    </button>
</a>

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  They describe characteristics of the cell nuclei present in the image.

Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

## Model Building
The DecisionTree class is a custom implementation of a decision tree algorithm for both classification and regression tasks.

- The `__init__` method initializes the decision tree with the type of task (classification or regression), the criterion for splitting nodes (entropy or gini), the minimum number of samples required at a leaf node, and the maximum depth of the tree.

- The `build_tree` method recursively builds the decision tree based on the provided dataset and current depth. It splits the dataset into features and target, checks the terminating conditions, and computes the best split. If the terminating conditions are met, it creates a new node with the best split values and recursively builds the left and right subtrees. If the terminating conditions are not met, it computes the leaf node value and returns a new node with that value.

- The `fit` method combines the features and target into a single dataset and builds the decision tree using that dataset.

- The `predict` method makes predictions for a given feature matrix by applying the `traverse_tree` method to each row of the feature matrix.

- The `traverse_tree` method traverses the decision tree to predict the target value for a given feature vector. It checks if the current node is a leaf node. If it is, it returns the value of the node. If it's not, it checks the feature value against the threshold of the node and recursively traverses either the left or right subtree.

- The `plot_tree` method prints the decision tree. It recursively traverses the tree and prints the feature, threshold, and value at each node.
