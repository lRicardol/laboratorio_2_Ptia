import pandas as pd
from main.Parte1.criterio import Entropy

X = pd.DataFrame({
    'A': ['sol', 'sol', 'lluvia', 'sol', 'lluvia'],
})
Y = pd.Series(['jugar', 'jugar', 'no_jugar', 'jugar', 'no_jugar'])

ent = Entropy()

print("Entropía total:", round(ent.impurity(Y), 4))
print("Ganancia de información (A):", round(ent.gain('A', X, Y), 4))
