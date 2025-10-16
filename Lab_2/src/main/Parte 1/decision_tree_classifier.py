import pandas as pd
from main.Parte1.decision_tree import DecisionTree
from main.Parte1.metricas import Accuracy, Precision, Recall, F1Score

# ================================================================
# CLASIFICADOR COMPLETO: Árbol de Decisión + Métricas
# ================================================================

class DecisionTreeClassifier:
    def __init__(self, criterion_name="entropy"):

        self.tree = DecisionTree(criterion_name)
        self.fitted = False

    def fit(self, X: pd.DataFrame, Y: pd.Series):

        self.tree.fit(X, Y)
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> pd.Series:

        if not self.fitted:
            raise Exception("El modelo no ha sido entrenado. Llama a fit() primero.")
        return self.tree.predict(X)

    def evaluate(self, X: pd.DataFrame, Y_true: pd.Series):

        if not self.fitted:
            raise Exception("El modelo no ha sido entrenado.")

        Y_pred = self.predict(X)

        metrics = {
            "Accuracy": Accuracy().value(Y_true, Y_pred),
            "Precision": Precision().value(Y_true, Y_pred),
            "Recall": Recall().value(Y_true, Y_pred),
            "F1": F1Score().value(Y_true, Y_pred)
        }

        return metrics

    def print_tree(self):

        if not self.fitted:
            print("El árbol aún no ha sido entrenado.")
        else:
            self.tree.print_tree()
