import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# ==========================================================
# CLASE BASE ABSTRACTA PARA CRITERIOS DE SELECCIÓN
# ==========================================================
class Criterium(ABC):

    @classmethod
    def use(cls, name: str):

        criteriums = {
            "entropy": Entropy
        }
        criterium_class = criteriums.get(name.lower())
        if criterium_class is None:
            raise ValueError(f"Criterio desconocido: {name}")
        return criterium_class()

    @abstractmethod
    def impurity(self, V: pd.Series) -> float:
        pass

    @abstractmethod
    def gain(self, a: str, X: pd.DataFrame, Y: pd.Series) -> float:
        pass

    @abstractmethod
    def treeImpurity(self, nodes: [pd.DataFrame]) -> float:
        pass


# ==========================================================
# IMPLEMENTACIÓN: ENTROPÍA
# ==========================================================

class Entropy(Criterium):

    def impurity(self, V: pd.Series) -> float:

        if len(V) == 0:
            return 0.0
        p = V.value_counts(normalize=True)
        return -np.sum(p * np.log2(p + 1e-9))  # +1e-9 evita log(0)

    def gain(self, a: str, X: pd.DataFrame, Y: pd.Series) -> float:

        total_entropy = self.impurity(Y)

        values = X[a].unique()
        weighted_entropy = 0.0
        for v in values:
            subset = Y[X[a] == v]
            weighted_entropy += (len(subset) / len(Y)) * self.impurity(subset)

        return total_entropy - weighted_entropy

    def treeImpurity(self, nodes: [pd.Series]) -> float:

        total = sum(len(n) for n in nodes)
        if total == 0:
            return 0.0
        return sum((len(n) / total) * self.impurity(n) for n in nodes)
