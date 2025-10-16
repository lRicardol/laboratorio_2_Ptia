import pandas as pd
from main.Parte1.metricas import Accuracy, Precision, Recall, F1Score

Y = pd.Series([1, 0, 1, 1, 0, 1])
Yp = pd.Series([1, 0, 1, 0, 0, 1])

print("Accuracy:", round(Accuracy().value(Y, Yp), 3))
print("Precision:", round(Precision().value(Y, Yp), 3))
print("Recall:", round(Recall().value(Y, Yp), 3))
print("F1:", round(F1Score().value(Y, Yp), 3))
