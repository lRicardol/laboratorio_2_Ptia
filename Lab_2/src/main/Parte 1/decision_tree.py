import pandas as pd
import numpy as np
from main.Parte1.criterio import Criterium

# ==========================================================
# CLASE PARA REPRESENTAR UN NODO DEL ÁRBOL
# ==========================================================
class Node:
    def __init__(self, attribute=None, value=None, children=None, label=None):

        self.attribute = attribute
        self.value = value
        self.children = children if children else {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


# ==========================================================
# CLASE PRINCIPAL DEL ÁRBOL DE DECISIÓN
# ==========================================================

class DecisionTree:
    def __init__(self, criterion_name="entropy"):

        self.criterion = Criterium.use(criterion_name)
        self.root = None

    def fit(self, X: pd.DataFrame, Y: pd.Series):
        self.root = self._build_tree(X, Y)

    def _build_tree(self, X: pd.DataFrame, Y: pd.Series) -> Node:

        if len(Y.unique()) == 1:
            return Node(label=Y.iloc[0])

        if X.empty:
            return Node(label=Y.mode()[0])

        gains = {a: self.criterion.gain(a, X, Y) for a in X.columns}
        best_attr = max(gains, key=gains.get)

        if gains[best_attr] <= 1e-6:
            return Node(label=Y.mode()[0])

        node = Node(attribute=best_attr)

        for v in X[best_attr].unique():
            mask = X[best_attr] == v
            sub_X = X[mask].drop(columns=[best_attr])
            sub_Y = Y[mask]

            if sub_Y.empty:
                child = Node(label=Y.mode()[0])
            else:
                child = self._build_tree(sub_X, sub_Y)

            node.children[v] = child

        return node

    def predict_instance(self, x, node=None):

        if node is None:
            node = self.root

        if node.is_leaf():
            return node.label
        value = x.get(node.attribute, None)

        if value not in node.children:
            child_labels = [child.label for child in node.children.values() if child.is_leaf()]
            if child_labels:
                return max(set(child_labels), key=child_labels.count)
            else:
                return None

        return self.predict_instance(x, node.children[value])

    def predict(self, X: pd.DataFrame):

        return X.apply(lambda row: self.predict_instance(row), axis=1)

    def print_tree(self, node=None, depth=0):

        if node is None:
            node = self.root

        if node.is_leaf():
            print("  " * depth + f"--> Clase: {node.label}")
        else:
            for v, child in node.children.items():
                print("  " * depth + f"[{node.attribute} = {v}]")
                self.print_tree(child, depth + 1)
