import pandas as pd
from abc import ABC, abstractmethod

# ============================================
# CLASE BASE ABSTRACTA PARA LAS MÉTRICAS
# ============================================
class Metric(ABC):

    @classmethod
    def use(cls, name: str):

        metrics = {
            "accuracy": Accuracy,
            "precision": Precision,
            "recall": Recall,
            "f1": F1Score
        }
        metric_class = metrics.get(name.lower())
        if metric_class is None:
            raise ValueError(f"Métrica desconocida: {name}")
        return metric_class()

    @abstractmethod
    def value(self, Y: pd.Series, Yp: pd.Series) -> float:
        pass


# ============================================
# MÉTRICA: EXACTITUD (Accuracy)
# ============================================
class Accuracy(Metric):
    def value(self, Y: pd.Series, Yp: pd.Series) -> float:
        return (Y == Yp).mean()


# ============================================
# MÉTRICA: PRECISIÓN (Precision)
# ============================================
class Precision(Metric):
    def value(self, Y: pd.Series, Yp: pd.Series) -> float:
        VP = ((Y == 1) & (Yp == 1)).sum()
        FP = ((Y == 0) & (Yp == 1)).sum()
        if (VP + FP) == 0:
            return 0.0
        return VP / (VP + FP)


# ============================================
# MÉTRICA: EXHAUSTIVIDAD (Recall)
# ============================================
class Recall(Metric):
    def value(self, Y: pd.Series, Yp: pd.Series) -> float:
        VP = ((Y == 1) & (Yp == 1)).sum()
        FN = ((Y == 1) & (Yp == 0)).sum()
        if (VP + FN) == 0:
            return 0.0
        return VP / (VP + FN)


# ============================================
# MÉTRICA: F1-SCORE
# ============================================
class F1Score(Metric):
    def value(self, Y: pd.Series, Yp: pd.Series) -> float:
        precision = Precision().value(Y, Yp)
        recall = Recall().value(Y, Yp)
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)