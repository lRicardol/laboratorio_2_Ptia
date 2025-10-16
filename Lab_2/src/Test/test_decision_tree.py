import pandas as pd
from main.Parte1.decision_tree import DecisionTree

X = pd.DataFrame({
    'Clima': ['sol', 'sol', 'lluvia', 'sol', 'lluvia'],
    'Temperatura': ['alta', 'alta', 'baja', 'media', 'baja']
})
Y = pd.Series(['jugar', 'jugar', 'no_jugar', 'jugar', 'no_jugar'])

tree = DecisionTree(criterion_name="entropy")
tree.fit(X, Y)

print("Árbol generado:")
tree.print_tree()

X_test = pd.DataFrame({
    'Clima': ['sol', 'lluvia'],
    'Temperatura': ['baja', 'alta']
})

print("\nPredicciones:")
print(tree.predict(X_test))
